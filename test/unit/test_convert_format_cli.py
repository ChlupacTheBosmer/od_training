import pytest

from src.od_training.dataset import convert_cli as mod

pytestmark = pytest.mark.unit


def test_convert_format_yolo2coco_requires_images():
    with pytest.raises(SystemExit) as exc:
        mod.main(["yolo2coco", "--input", "labels", "--output", "out.json"])

    assert exc.value.code == 1


def test_convert_format_yolo2coco_dispatches(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        mod,
        "convert_yolo_to_coco",
        lambda inp, images, out: calls.update({"args": (inp, images, out)}),
    )

    mod.main(
        [
            "yolo2coco",
            "--input",
            "labels",
            "--images",
            "images",
            "--output",
            "out.json",
        ]
    )
    assert calls["args"] == ("labels", "images", "out.json")


def test_convert_format_coco2yolo_dispatches(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        mod,
        "convert_coco_to_yolo",
        lambda inp, out: calls.update({"args": (inp, out)}),
    )

    mod.main(["coco2yolo", "--input", "ann_dir", "--output", "yolo_out"])
    assert calls["args"] == ("ann_dir", "yolo_out")
