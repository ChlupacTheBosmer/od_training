import sys

import pytest


pytestmark = pytest.mark.unit


def test_convert_format_yolo2coco_requires_images(load_script_module, monkeypatch):
    mod = load_script_module("convert_format.py")

    monkeypatch.setattr(
        sys,
        "argv",
        ["convert_format.py", "yolo2coco", "--input", "labels", "--output", "out.json"],
    )

    with pytest.raises(SystemExit) as exc:
        mod.main()

    assert exc.value.code == 1


def test_convert_format_yolo2coco_dispatches(load_script_module, monkeypatch):
    mod = load_script_module("convert_format.py")

    calls = {}

    monkeypatch.setattr(
        mod,
        "convert_yolo_to_coco",
        lambda inp, images, out: calls.update({"args": (inp, images, out)}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_format.py",
            "yolo2coco",
            "--input",
            "labels",
            "--images",
            "images",
            "--output",
            "out.json",
        ],
    )

    mod.main()
    assert calls["args"] == ("labels", "images", "out.json")


def test_convert_format_coco2yolo_dispatches(load_script_module, monkeypatch):
    mod = load_script_module("convert_format.py")

    calls = {}

    monkeypatch.setattr(
        mod,
        "convert_coco_to_yolo",
        lambda inp, out: calls.update({"args": (inp, out)}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_format.py",
            "coco2yolo",
            "--input",
            "ann_dir",
            "--output",
            "yolo_out",
        ],
    )

    mod.main()
    assert calls["args"] == ("ann_dir", "yolo_out")
