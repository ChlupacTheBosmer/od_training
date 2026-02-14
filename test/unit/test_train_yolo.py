import importlib
import sys
import types

import pytest


pytestmark = pytest.mark.unit


class _FakeModel:
    def __init__(self):
        self.train_args = None
        self.export_calls = []

    def train(self, **kwargs):
        self.train_args = kwargs
        return {"ok": True}

    def export(self, **kwargs):
        self.export_calls.append(kwargs)


@pytest.fixture
def train_yolo_module(monkeypatch):
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = object

    fake_clearml = types.ModuleType("clearml")
    fake_clearml.Task = types.SimpleNamespace(init=lambda **_: object())

    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setitem(sys.modules, "clearml", fake_clearml)

    mod = importlib.import_module("od_training.train.yolo")
    return importlib.reload(mod)


def test_train_yolo_sets_musgd_for_yolo26(train_yolo_module, monkeypatch):
    mod = train_yolo_module

    created = {}

    def fake_yolo(_):
        model = _FakeModel()
        created["model"] = model
        return model

    monkeypatch.setattr(mod, "YOLO", fake_yolo)
    monkeypatch.setattr(mod.Task, "init", lambda **_: object())

    mod.train_yolo(
        "yolo26n.pt",
        "data.yaml",
        "proj",
        "exp",
        epochs=2,
        device="cuda",
    )

    model = created["model"]
    assert model.train_args["optimizer"] == "MuSGD"
    assert model.export_calls[0]["format"] == "engine"
    assert model.export_calls[0]["end2end"] is True


def test_train_yolo_preserves_explicit_optimizer(train_yolo_module, monkeypatch):
    mod = train_yolo_module

    created = {}

    def fake_yolo(_):
        model = _FakeModel()
        created["model"] = model
        return model

    monkeypatch.setattr(mod, "YOLO", fake_yolo)
    monkeypatch.setattr(mod.Task, "init", lambda **_: object())

    mod.train_yolo(
        "yolo26n.pt",
        "data.yaml",
        "proj",
        "exp",
        epochs=2,
        device="cuda",
        optimizer="Adam",
    )

    assert created["model"].train_args["optimizer"] == "Adam"


def test_train_yolo_skips_export_on_cpu(train_yolo_module, monkeypatch):
    mod = train_yolo_module

    created = {}

    def fake_yolo(_):
        model = _FakeModel()
        created["model"] = model
        return model

    monkeypatch.setattr(mod, "YOLO", fake_yolo)
    monkeypatch.setattr(mod.Task, "init", lambda **_: object())

    mod.train_yolo(
        "yolo11n.pt",
        "data.yaml",
        "proj",
        "exp",
        epochs=5,
        device="cpu",
    )

    assert created["model"].export_calls == []


def test_train_yolo_continues_when_clearml_init_fails(train_yolo_module, monkeypatch):
    mod = train_yolo_module

    created = {}

    def fake_yolo(_):
        model = _FakeModel()
        created["model"] = model
        return model

    monkeypatch.setattr(mod, "YOLO", fake_yolo)

    def fail_task_init(**_):
        raise RuntimeError("ClearML unavailable")

    monkeypatch.setattr(mod.Task, "init", fail_task_init)

    result = mod.train_yolo(
        "yolo11n.pt",
        "data.yaml",
        "proj",
        "exp",
        epochs=1,
        batch=1,
        device="cpu",
    )

    assert result == {"ok": True}
    assert created["model"].train_args["data"] == "data.yaml"


def test_train_yolo_calls_preflight_when_enabled(train_yolo_module, monkeypatch):
    mod = train_yolo_module
    created = {}
    seen = {}

    def fake_yolo(_):
        model = _FakeModel()
        created["model"] = model
        return model

    def fake_preflight(*, data_yaml, fail_on_warnings=False):
        seen["data_yaml"] = data_yaml
        seen["fail_on_warnings"] = fail_on_warnings
        return {"ok": True, "checked_splits": ["train"], "warnings": []}

    monkeypatch.setattr(mod, "YOLO", fake_yolo)
    monkeypatch.setattr(mod.Task, "init", lambda **_: object())
    monkeypatch.setattr(mod, "validate_yolo_training_inputs", fake_preflight)

    mod.train_yolo(
        "yolo11n.pt",
        "data.yaml",
        "proj",
        "exp",
        validate_data=True,
        fail_on_validation_warnings=True,
        epochs=1,
        batch=1,
        device="cpu",
    )

    assert seen["data_yaml"] == "data.yaml"
    assert seen["fail_on_warnings"] is True
