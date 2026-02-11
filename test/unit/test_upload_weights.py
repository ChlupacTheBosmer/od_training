import builtins
import sys
import types

import pytest

from src.od_training.utility import roboflow_upload as mod


pytestmark = pytest.mark.unit


def test_upload_weights_missing_file():
    with pytest.raises(FileNotFoundError):
        mod.upload_weights(
            api_key="key",
            workspace="ws",
            project_name="proj",
            version_num=1,
            weights_path="/does/not/exist.pt",
        )


def test_upload_weights_empty_file(tmp_path):
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")

    with pytest.raises(ValueError):
        mod.upload_weights(
            api_key="key",
            workspace="ws",
            project_name="proj",
            version_num=1,
            weights_path=str(empty),
        )


def test_upload_weights_import_error_for_roboflow(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"abc")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "roboflow":
            raise ImportError("roboflow not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        mod.upload_weights(
            api_key="key",
            workspace="ws",
            project_name="proj",
            version_num=1,
            weights_path=str(weights),
        )


def test_upload_weights_success(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"abc")

    calls = {}

    class FakeVersion:
        def deploy(self, model_type, model_path):
            calls["deploy"] = (model_type, model_path)

    class FakeProject:
        def version(self, version_num):
            calls["version"] = version_num
            return FakeVersion()

    class FakeWorkspace:
        def project(self, project_name):
            calls["project"] = project_name
            return FakeProject()

    class FakeRoboflow:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        def workspace(self, workspace_name):
            calls["workspace"] = workspace_name
            return FakeWorkspace()

    fake_module = types.ModuleType("roboflow")
    fake_module.Roboflow = FakeRoboflow
    monkeypatch.setitem(sys.modules, "roboflow", fake_module)

    mod.upload_weights(
        api_key="key123",
        workspace="ws",
        project_name="proj",
        version_num=4,
        weights_path=str(weights),
        model_type="rf-detr",
    )

    assert calls["api_key"] == "key123"
    assert calls["workspace"] == "ws"
    assert calls["project"] == "proj"
    assert calls["version"] == 4
    assert calls["deploy"] == ("rf-detr", str(weights))


def test_upload_weights_deploy_failure_propagates(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"abc")

    class FakeVersion:
        def deploy(self, model_type, model_path):
            raise RuntimeError("deploy failed")

    class FakeProject:
        def version(self, version_num):
            return FakeVersion()

    class FakeWorkspace:
        def project(self, project_name):
            return FakeProject()

    class FakeRoboflow:
        def __init__(self, api_key):
            pass

        def workspace(self, workspace_name):
            return FakeWorkspace()

    fake_module = types.ModuleType("roboflow")
    fake_module.Roboflow = FakeRoboflow
    monkeypatch.setitem(sys.modules, "roboflow", fake_module)

    with pytest.raises(RuntimeError):
        mod.upload_weights(
            api_key="key",
            workspace="ws",
            project_name="proj",
            version_num=1,
            weights_path=str(weights),
        )
