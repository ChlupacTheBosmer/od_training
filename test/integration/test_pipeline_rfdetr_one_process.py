import json
import shutil
import importlib
import uuid
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow]


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

        # RF-DETR convention used in this repo scripts/examples
        shutil.copy2(ann_src, split_dir / "_annotations.coco.json")


def test_rfdetr_one_process_export_train_infer(
    require_model_smoke,
    make_yolo_dataset,
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("rfdetr")
    fo = pytest.importorskip("fiftyone")
    dataset_manager = importlib.import_module("src.od_training.dataset.manager")
    train_rfdetr_mod = importlib.import_module("src.od_training.train.rfdetr")
    inference_mod = importlib.import_module("src.od_training.infer.runner")

    runtime_tmp = tmp_path / "runtime_tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)

    # Keep smoke tests self-contained and stable under constrained local temp space.
    monkeypatch.setenv("TMPDIR", str(runtime_tmp))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    monkeypatch.setenv("MPLCONFIGDIR", str(runtime_tmp / "mpl"))

    monkeypatch.setattr(train_rfdetr_mod.Task, "init", lambda **_: object())

    dataset_name = f"it_rfdetr_{uuid.uuid4().hex[:8]}"

    dataset_dir = make_yolo_dataset(tmp_path, image_count=1)
    export_dir = tmp_path / "exported"
    rfdetr_dataset_dir = tmp_path / "rfdetr_dataset"
    output_dir = tmp_path / "rf_output"
    infer_dir = tmp_path / "infer_out"

    try:
        ds = dataset_manager.load_or_create_dataset(str(dataset_dir / "data.yaml"), dataset_name)
        sample = ds.first()
        sample.tags = ["train", "val", "test"]
        sample.save()

        classes = ds.default_classes or ds.distinct("ground_truth.detections.label")
        dataset_manager.export_pipeline(
            ds,
            str(export_dir),
            classes=classes,
            copy_images=True,
        )

        _materialize_rfdetr_dataset(export_dir, rfdetr_dataset_dir)

        # Ensure all expected splits exist for RF-DETR training script contract
        for split in ["train", "valid", "test"]:
            split_dir = rfdetr_dataset_dir / split
            if not split_dir.exists():
                pytest.skip(f"Missing split for RF-DETR training: {split_dir}")

        train_rfdetr_mod.train_rfdetr(
            dataset_dir=str(rfdetr_dataset_dir),
            model_type="rfdetr_nano",
            epochs=1,
            batch_size=1,
            lr=1e-4,
            project_name="RFDETR_Smoke",
            exp_name="rf_smoke",
            device="cpu",
            output_dir=str(output_dir),
            tensorboard=False,
            wandb=False,
            num_workers=0,
            run_test=False,
        )

        checkpoints = list(output_dir.rglob("*.pth"))
        assert checkpoints, "No RF-DETR checkpoint produced"
        checkpoint = checkpoints[0]

        source_dir = rfdetr_dataset_dir / "test"
        inference_mod.run_inference(
            source=str(source_dir),
            model_name=str(checkpoint),
            model_type="rfdetr",
            rfdetr_arch="rfdetr_nano",
            save_dir=str(infer_dir),
        )

        outputs = list(infer_dir.glob("*_annotated.jpg"))
        assert outputs, "No annotated outputs were produced"
        assert all(p.stat().st_size > 0 for p in outputs)

    finally:
        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)
