import json
from pathlib import Path

import pytest
from PIL import Image

from od_training.train.preflight import (
    validate_rfdetr_training_inputs,
    validate_yolo_training_inputs,
)


pytestmark = pytest.mark.unit


def _write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color="red").save(path)


def _write_coco(path: Path, *, empty_categories: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 16, "height": 16}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
        "categories": [] if empty_categories else [{"id": 1, "name": "obj"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_yolo_training_inputs_success(tmp_path):
    ds = tmp_path / "yolo_ds"
    _write_image(ds / "images" / "train" / "a.jpg")
    (ds / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (ds / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    data_yaml = ds / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nnames: ['obj']\n",
        encoding="utf-8",
    )

    result = validate_yolo_training_inputs(data_yaml=str(data_yaml))
    assert result["ok"] is True
    assert "train" in result["checked_splits"]


def test_validate_yolo_training_inputs_raises_on_missing_labels_dir(tmp_path):
    ds = tmp_path / "yolo_ds"
    _write_image(ds / "images" / "train" / "a.jpg")

    data_yaml = ds / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nnames: ['obj']\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        validate_yolo_training_inputs(data_yaml=str(data_yaml))


def test_validate_rfdetr_training_inputs_success(tmp_path):
    root = tmp_path / "rfdetr_ds"
    for split in ("train", "valid", "test"):
        _write_image(root / split / "a.jpg")
        _write_coco(root / split / "_annotations.coco.json")

    result = validate_rfdetr_training_inputs(dataset_dir=str(root))
    assert result["ok"] is True
    assert set(result["splits"].keys()) == {"train", "valid", "test"}


def test_validate_rfdetr_training_inputs_fail_on_warnings(tmp_path):
    root = tmp_path / "rfdetr_ds"
    for split in ("train", "valid", "test"):
        _write_image(root / split / "a.jpg")
        _write_coco(root / split / "_annotations.coco.json", empty_categories=True)

    with pytest.raises(ValueError):
        validate_rfdetr_training_inputs(dataset_dir=str(root), fail_on_warnings=True)

