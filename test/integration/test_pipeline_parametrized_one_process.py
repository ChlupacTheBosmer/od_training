import importlib
import shutil
import sys
import types
import uuid
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration]


def _stub_train_and_inference_dependencies(monkeypatch):
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = object

    fake_clearml = types.ModuleType("clearml")
    fake_clearml.Task = types.SimpleNamespace(init=lambda **_: object())

    fake_supervision = types.ModuleType("supervision")
    fake_supervision.BoxAnnotator = type(
        "BoxAnnotator",
        (),
        {"annotate": lambda self, scene, detections: scene},
    )
    fake_supervision.LabelAnnotator = type(
        "LabelAnnotator",
        (),
        {"annotate": lambda self, scene, detections: scene},
    )
    fake_supervision.Detections = type(
        "Detections",
        (),
        {"from_ultralytics": staticmethod(lambda _: {})},
    )
    fake_supervision.get_video_frames_generator = lambda _source: iter([])

    fake_rfdetr = types.ModuleType("rfdetr")
    for cls_name in [
        "RFDETRNano",
        "RFDETRSmall",
        "RFDETRMedium",
        "RFDETRLarge",
        "RFDETRXLarge",
        "RFDETR2XLarge",
        "RFDETRBase",
    ]:
        setattr(fake_rfdetr, cls_name, type(cls_name, (), {}))

    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setitem(sys.modules, "clearml", fake_clearml)
    monkeypatch.setitem(sys.modules, "supervision", fake_supervision)
    monkeypatch.setitem(sys.modules, "rfdetr", fake_rfdetr)


def _materialize_rfdetr_dataset(export_dir: Path, target_dir: Path):
    split_map = {
        "train": "train",
        "val": "valid",
        "test": "test",
    }

    for export_split, rf_split in split_map.items():
        images_src = export_dir / "images" / export_split
        ann_src = export_dir / f"_annotations_{export_split}.coco.json"
        if not images_src.exists() or not ann_src.exists():
            continue

        split_dir = target_dir / rf_split
        split_dir.mkdir(parents=True, exist_ok=True)

        for image in images_src.iterdir():
            if image.is_file() or image.is_symlink():
                shutil.copy2(image.resolve() if image.is_symlink() else image, split_dir / image.name)

        shutil.copy2(ann_src, split_dir / "_annotations.coco.json")


def _import_and_reload(module_path: str):
    mod = importlib.import_module(module_path)
    return importlib.reload(mod)


@pytest.mark.parametrize("model_type", ["yolo", "rfdetr"])
def test_one_process_pipeline_contract(
    model_type,
    make_yolo_dataset,
    tmp_path,
    monkeypatch,
):
    fo = pytest.importorskip("fiftyone")
    _stub_train_and_inference_dependencies(monkeypatch)

    dataset_manager = _import_and_reload("src.od_training.dataset.manager")
    train_yolo_mod = _import_and_reload("src.od_training.train.yolo")
    train_rfdetr_mod = _import_and_reload("src.od_training.train.rfdetr")
    inference_mod = _import_and_reload("src.od_training.infer.runner")

    dataset_name = f"it_param_{model_type}_{uuid.uuid4().hex[:8]}"

    dataset_dir = make_yolo_dataset(tmp_path, image_count=1)
    export_dir = tmp_path / f"export_{model_type}"
    infer_dir = tmp_path / f"infer_{model_type}"

    calls = []

    def fake_train_yolo(*args, **kwargs):
        calls.append("train_yolo")
        weights_dir = tmp_path / "yolo_runs" / "exp" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"x")
        return {"ok": True}

    def fake_train_rfdetr(*args, **kwargs):
        calls.append("train_rfdetr")
        out = Path(kwargs.get("output_dir", tmp_path / "rf_out"))
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoint.pth").write_bytes(b"x")

    def fake_run_inference(*args, **kwargs):
        calls.append("run_inference")
        out = Path(kwargs["save_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_annotated.jpg").write_bytes(b"x")

    monkeypatch.setattr(train_yolo_mod, "train_yolo", fake_train_yolo)
    monkeypatch.setattr(train_rfdetr_mod, "train_rfdetr", fake_train_rfdetr)
    monkeypatch.setattr(inference_mod, "run_inference", fake_run_inference)

    try:
        ds = dataset_manager.load_or_create_dataset(str(dataset_dir / "data.yaml"), dataset_name)
        sample = ds.first()
        sample.tags = ["train", "val", "test"]
        sample.save()

        classes = ds.default_classes or ds.distinct("ground_truth.detections.label")
        dataset_manager.export_pipeline(ds, str(export_dir), classes=classes, copy_images=True)

        if model_type == "yolo":
            data_yaml = export_dir / "dataset.yaml"
            train_yolo_mod.train_yolo(
                model_name="dummy.pt",
                data_yaml=str(data_yaml),
                project_name=str(tmp_path / "yolo_runs"),
                exp_name="exp",
                epochs=1,
                batch=1,
                imgsz=64,
                device="cpu",
            )
            inference_mod.run_inference(
                source=str(export_dir / "images" / "train"),
                model_name=str(tmp_path / "yolo_runs" / "exp" / "weights" / "best.pt"),
                model_type="yolo",
                save_dir=str(infer_dir),
            )
            assert calls == ["train_yolo", "run_inference"]
        else:
            rf_dir = tmp_path / "rf_ds"
            _materialize_rfdetr_dataset(export_dir, rf_dir)

            out_dir = tmp_path / "rf_out"
            train_rfdetr_mod.train_rfdetr(
                dataset_dir=str(rf_dir),
                model_type="rfdetr_nano",
                epochs=1,
                batch_size=1,
                lr=1e-4,
                project_name="p",
                exp_name="e",
                device="cpu",
                output_dir=str(out_dir),
                tensorboard=False,
                wandb=False,
            )
            inference_mod.run_inference(
                source=str(rf_dir / "test"),
                model_name=str(out_dir / "checkpoint.pth"),
                model_type="rfdetr",
                rfdetr_arch="rfdetr_nano",
                save_dir=str(infer_dir),
            )
            assert calls == ["train_rfdetr", "run_inference"]

        assert (infer_dir / "frame_annotated.jpg").exists()

    finally:
        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)
