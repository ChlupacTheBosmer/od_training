import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def require_model_smoke():
    if os.environ.get("RUN_MODEL_SMOKE") != "1":
        pytest.skip("Set RUN_MODEL_SMOKE=1 to run model training/inference smoke tests")


@pytest.fixture
def write_test_image():
    def _write(path: Path, size=(640, 640), color=(0, 0, 255)):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        image[:, :] = color
        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise RuntimeError(f"Failed to write test image to {path}")
        return path

    return _write


@pytest.fixture
def make_yolo_dataset(write_test_image):
    def _make(base_dir: Path, dataset_name: str = "dummy_yolo", image_count: int = 1):
        dataset_dir = base_dir / dataset_name
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for i in range(image_count):
            stem = f"img_{i}"
            write_test_image(images_dir / f"{stem}.jpg")
            (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        data_yaml = {
            "path": str(dataset_dir.resolve()),
            "train": "images",
            "val": "images",
            "names": {0: "object"},
        }
        (dataset_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml), encoding="utf-8")
        return dataset_dir

    return _make


@pytest.fixture
def make_coco_dataset(write_test_image):
    def _make(base_dir: Path, dataset_name: str = "dummy_coco"):
        import json

        root = base_dir / dataset_name
        for split in ["train", "valid", "test"]:
            split_dir = root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            write_test_image(split_dir / "test.jpg")

            ann = {
                "images": [
                    {"id": 1, "file_name": "test.jpg", "width": 640, "height": 640},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 0,
                        "bbox": [256, 256, 128, 128],
                        "area": 16384,
                        "iscrowd": 0,
                    },
                ],
                "categories": [{"id": 0, "name": "object"}],
            }
            (split_dir / "_annotations.coco.json").write_text(json.dumps(ann), encoding="utf-8")

        return root

    return _make
