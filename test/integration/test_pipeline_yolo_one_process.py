import uuid
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_yolo_one_process_export_train_infer(
    require_model_smoke,
    load_script_module,
    make_yolo_dataset,
    monkeypatch,
    repo_root,
    tmp_path,
):
    pytest.importorskip("ultralytics")
    fo = pytest.importorskip("fiftyone")

    runtime_tmp = tmp_path / "runtime_tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("TMPDIR", str(runtime_tmp))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    monkeypatch.setenv("MPLCONFIGDIR", str(runtime_tmp / "mpl"))

    dataset_manager = load_script_module("dataset_manager.py")
    train_yolo_mod = load_script_module("train_yolo.py")
    inference_mod = load_script_module("inference.py")
    monkeypatch.setattr(train_yolo_mod.Task, "init", lambda **_: object())

    local_weights = repo_root / "yolo11n.pt"
    if not local_weights.exists():
        pytest.skip(f"Missing local YOLO weights: {local_weights}")

    dataset_name = f"it_yolo_{uuid.uuid4().hex[:8]}"

    dataset_dir = make_yolo_dataset(tmp_path, image_count=1)
    export_dir = tmp_path / "exported"
    train_project = tmp_path / "train_runs"
    infer_dir = tmp_path / "infer_out"

    try:
        ds = dataset_manager.load_or_create_dataset(str(dataset_dir / "data.yaml"), dataset_name)
        sample = ds.first()
        sample.tags = ["train", "val"]
        sample.save()

        classes = ds.default_classes or ds.distinct("ground_truth.detections.label")
        dataset_manager.export_pipeline(
            ds,
            str(export_dir),
            classes=classes,
            copy_images=True,
        )

        data_yaml = export_dir / "dataset.yaml"
        assert data_yaml.exists()

        train_yolo_mod.train_yolo(
            model_name=str(local_weights),
            data_yaml=str(data_yaml),
            project_name=str(train_project),
            exp_name="yolo_smoke",
            epochs=1,
            batch=1,
            imgsz=64,
            device="cpu",
        )

        best = train_project / "yolo_smoke" / "weights" / "best.pt"
        last = train_project / "yolo_smoke" / "weights" / "last.pt"
        weights = best if best.exists() else last
        assert weights.exists(), "No YOLO checkpoint found after training"

        source_dir = export_dir / "images" / "val"
        assert source_dir.exists()

        inference_mod.run_inference(
            source=str(source_dir),
            model_name=str(weights),
            model_type="yolo",
            save_dir=str(infer_dir),
        )

        outputs = list(infer_dir.glob("*_annotated.jpg"))
        assert outputs, "No annotated outputs were produced"
        assert all(p.stat().st_size > 0 for p in outputs)

    finally:
        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)
