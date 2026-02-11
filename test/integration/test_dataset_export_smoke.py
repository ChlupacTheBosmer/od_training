import uuid
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration]


def test_dataset_export_pipeline_smoke(load_script_module, make_yolo_dataset, tmp_path):
    fo = pytest.importorskip("fiftyone")
    mod = load_script_module("dataset_manager.py")

    dataset_name = f"it_export_{uuid.uuid4().hex[:8]}"
    aug_name = f"it_export_aug_{uuid.uuid4().hex[:8]}"

    dataset_dir = make_yolo_dataset(tmp_path, image_count=1)
    export_dir = tmp_path / "exported"

    try:
        ds = mod.load_or_create_dataset(str(dataset_dir / "data.yaml"), dataset_name)

        # Ensure deterministic split tags for this tiny fixture
        s = ds.first()
        s.tags = ["train"]
        s.save()

        aug_ds = mod.augment_samples(
            ds,
            filter_tags=["train"],
            new_dataset_name=aug_name,
            num_aug=1,
        )

        classes = aug_ds.default_classes or aug_ds.distinct("ground_truth.detections.label")
        mod.export_pipeline(
            aug_ds,
            str(export_dir),
            classes=classes,
            export_tags=["augmented"],
            copy_images=False,
        )

        dataset_yaml = export_dir / "dataset.yaml"
        image_train_dir = export_dir / "images" / "train"
        label_train_dir = export_dir / "labels" / "train"
        coco_json = export_dir / "_annotations_train.coco.json"

        assert dataset_yaml.exists()
        assert image_train_dir.exists()
        assert any(p.is_symlink() for p in image_train_dir.iterdir())

        label_files = list(label_train_dir.glob("*.txt"))
        assert label_files

        for lf in label_files:
            for line in lf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    assert len(line.split()) == 5

        assert coco_json.exists()
        assert coco_json.stat().st_size > 0

    finally:
        for name in [dataset_name, aug_name]:
            if name in fo.list_datasets():
                fo.delete_dataset(name)
