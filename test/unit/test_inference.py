import importlib
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest


pytestmark = pytest.mark.unit


def _make_fake_supervision_module():
    mod = types.ModuleType("supervision")

    class _BoxAnnotator:
        def annotate(self, scene, detections):
            return scene

    class _LabelAnnotator:
        def annotate(self, scene, detections):
            return scene

    class _Detections:
        @staticmethod
        def from_ultralytics(_):
            return {"source": "ultralytics"}

    mod.BoxAnnotator = _BoxAnnotator
    mod.LabelAnnotator = _LabelAnnotator
    mod.Detections = _Detections
    mod.get_video_frames_generator = lambda _source: iter([])
    return mod


def _make_fake_rfdetr_module():
    mod = types.ModuleType("rfdetr")

    def make_cls(name):
        class _Cls:
            def __init__(self, *args, **kwargs):
                self.name = name
                self.args = args
                self.kwargs = kwargs

            def predict(self, _frame, threshold=0.3):
                return [{"threshold": threshold}]

        _Cls.__name__ = name
        return _Cls

    mod.RFDETRNano = make_cls("RFDETRNano")
    mod.RFDETRSmall = make_cls("RFDETRSmall")
    mod.RFDETRMedium = make_cls("RFDETRMedium")
    mod.RFDETRLarge = make_cls("RFDETRLarge")
    mod.RFDETRXLarge = make_cls("RFDETRXLarge")
    mod.RFDETR2XLarge = make_cls("RFDETR2XLarge")
    mod.RFDETRBase = make_cls("RFDETRBase")
    return mod


@pytest.fixture
def inference_module(monkeypatch):
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = object

    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setitem(sys.modules, "supervision", _make_fake_supervision_module())
    monkeypatch.setitem(sys.modules, "rfdetr", _make_fake_rfdetr_module())

    mod = importlib.import_module("od_training.infer.runner")
    return importlib.reload(mod)


def test_load_rfdetr_model_known_key(inference_module):
    mod = inference_module

    model = mod.load_rfdetr_model("rfdetr_nano")
    assert model.__class__.__name__ == "RFDETRNano"


def test_load_rfdetr_model_missing_path_raises(inference_module):
    mod = inference_module

    with pytest.raises(FileNotFoundError):
        mod.load_rfdetr_model("/path/that/does/not/exist.pt")


def test_run_inference_rejects_unsupported_model_type(inference_module):
    mod = inference_module

    with pytest.raises(ValueError):
        mod.run_inference(
            source="some_source.jpg",
            model_name="model.pt",
            model_type="unknown",
        )


def test_run_inference_yolo_image_file_saves_output(inference_module, tmp_path):
    mod = inference_module

    image_path = tmp_path / "frame.jpg"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    class FakeYOLO:
        def __init__(self, _name):
            pass

        def predict(self, _frame, conf=0.3, iou=0.5, verbose=False):
            return [object()]

    mod.YOLO = FakeYOLO

    save_dir = tmp_path / "out"
    mod.run_inference(
        source=str(image_path),
        model_name="dummy.pt",
        model_type="yolo",
        save_dir=str(save_dir),
    )

    assert (save_dir / "frame_annotated.jpg").exists()


def test_run_inference_yolo_directory_preserves_source_names(inference_module, tmp_path):
    mod = inference_module

    source_dir = tmp_path / "images"
    source_dir.mkdir()

    for stem in ["a", "b"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        assert cv2.imwrite(str(source_dir / f"{stem}.jpg"), image)

    class FakeYOLO:
        def __init__(self, _name):
            pass

        def predict(self, _frame, conf=0.3, iou=0.5, verbose=False):
            return [object()]

    mod.YOLO = FakeYOLO

    save_dir = tmp_path / "out"
    mod.run_inference(
        source=str(source_dir),
        model_name="dummy.pt",
        model_type="yolo",
        save_dir=str(save_dir),
    )

    assert (save_dir / "a_annotated.jpg").exists()
    assert (save_dir / "b_annotated.jpg").exists()


def test_run_inference_rfdetr_path_works(inference_module, tmp_path):
    mod = inference_module

    image_path = tmp_path / "frame.jpg"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    class FakeRFModel:
        def predict(self, _frame, threshold=0.3):
            return [{"threshold": threshold}]

    mod.load_rfdetr_model = lambda *_args, **_kwargs: FakeRFModel()

    save_dir = tmp_path / "out"
    mod.run_inference(
        source=str(image_path),
        model_name="weights.pt",
        model_type="rfdetr",
        rfdetr_arch="rfdetr_nano",
        save_dir=str(save_dir),
    )

    assert (save_dir / "frame_annotated.jpg").exists()
