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
def train_yolo_module(load_script_module, monkeypatch):
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = object

    fake_clearml = types.ModuleType("clearml")
    fake_clearml.Task = types.SimpleNamespace(init=lambda **_: object())

    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setitem(sys.modules, "clearml", fake_clearml)

    return load_script_module("train_yolo.py")


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
