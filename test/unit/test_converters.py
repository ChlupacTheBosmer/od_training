import builtins
import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

import src.converters as converters


pytestmark = pytest.mark.unit


def test_convert_yolo_to_coco_calls_globox(monkeypatch, tmp_path):
    calls = {}

    class FakeAnnotations:
        def __len__(self):
            return 2

        def save_coco(self, output_path, auto_ids=False):
            calls["save_coco"] = (output_path, auto_ids)

    class FakeAnnotationSet:
        @staticmethod
        def from_yolo_v5(folder, image_folder):
            calls["from_yolo_v5"] = (folder, image_folder)
            return FakeAnnotations()

    monkeypatch.setattr(converters, "AnnotationSet", FakeAnnotationSet)

    out_json = tmp_path / "out.json"
    converters.convert_yolo_to_coco("labels", "images", str(out_json))

    assert calls["from_yolo_v5"] == (Path("labels"), Path("images"))
    assert calls["save_coco"] == (out_json, True)


def test_convert_coco_to_yolo_uses_ultralytics_when_available(monkeypatch):
    calls = {}

    converter_mod = types.ModuleType("ultralytics.data.converter")

    def fake_convert_coco(**kwargs):
        calls["kwargs"] = kwargs

    converter_mod.convert_coco = fake_convert_coco

    data_mod = types.ModuleType("ultralytics.data")
    data_mod.converter = converter_mod

    ultra_mod = types.ModuleType("ultralytics")
    ultra_mod.data = data_mod

    monkeypatch.setitem(sys.modules, "ultralytics", ultra_mod)
    monkeypatch.setitem(sys.modules, "ultralytics.data", data_mod)
    monkeypatch.setitem(sys.modules, "ultralytics.data.converter", converter_mod)

    converters.convert_coco_to_yolo("in_dir", "out_dir", use_segments=True)

    assert calls["kwargs"]["labels_dir"] == "in_dir"
    assert calls["kwargs"]["save_dir"] == "out_dir"
    assert calls["kwargs"]["use_segments"] is True


def test_convert_coco_to_yolo_falls_back_to_globox(monkeypatch, tmp_path):
    calls = []

    ann_json = tmp_path / "ann.json"
    ann_json.write_text("{}", encoding="utf-8")

    class FakeAnnotations:
        def save_yolo_v5(self, save_dir, label_to_id=None):
            calls.append(("save", save_dir, label_to_id))

    class FakeAnnotationSet:
        @staticmethod
        def from_coco(path):
            calls.append(("from_coco", Path(path)))
            return FakeAnnotations()

    monkeypatch.setattr(converters, "AnnotationSet", FakeAnnotationSet)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ultralytics.data.converter":
            raise ImportError("converter missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    out_dir = tmp_path / "yolo_out"
    converters.convert_coco_to_yolo(str(ann_json), str(out_dir))

    assert ("from_coco", ann_json) in calls
    assert any(call[0] == "save" and call[1] == out_dir for call in calls)


def test_validate_dataset_happy_path(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    Image.new("RGB", (32, 32), color="red").save(image_dir / "a.jpg")
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = converters.validate_dataset(str(image_dir), str(label_dir), class_names=["obj"])

    assert report.ok is True
    assert report.errors == []


def test_validate_dataset_warns_on_missing_label(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    Image.new("RGB", (32, 32), color="red").save(image_dir / "a.jpg")

    report = converters.validate_dataset(str(image_dir), str(label_dir))

    assert report.ok is True
    assert len(report.warnings) == 1


def test_validate_dataset_accepts_six_column_labels_with_warning(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    Image.new("RGB", (32, 32), color="red").save(image_dir / "a.jpg")
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")

    report = converters.validate_dataset(str(image_dir), str(label_dir), class_names=["obj"])

    assert report.ok is True
    assert any("6-column" in message for _, message in report.warnings)


def test_validate_dataset_flags_invalid_values(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    Image.new("RGB", (32, 32), color="red").save(image_dir / "a.jpg")
    (label_dir / "a.txt").write_text("5 1.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = converters.validate_dataset(str(image_dir), str(label_dir), class_names=["obj"])

    assert report.ok is False
    assert len(report.errors) >= 2  # coord out of range + class id out of range


def test_fix_json_label_non_numeric(tmp_path):
    # Quick coverage for non-numeric branch inside _validate_yolo_label via validate_dataset
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    Image.new("RGB", (32, 32), color="red").save(image_dir / "a.jpg")
    (label_dir / "a.txt").write_text("0 0.5 bad 0.2 0.2\n", encoding="utf-8")

    report = converters.validate_dataset(str(image_dir), str(label_dir), class_names=["obj"])
    assert report.ok is False
    assert any("non-numeric" in message for _, message in report.errors)
