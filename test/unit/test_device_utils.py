import builtins
import sys
import types

import pytest

from src.device_utils import resolve_device


pytestmark = pytest.mark.unit


def test_resolve_device_explicit_wins(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_uses_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert resolve_device(None) == "cuda"


def test_resolve_device_uses_torch_cuda_probe(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _: "Fake GPU",
        )
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_device(None) == "cuda"


def test_resolve_device_falls_back_to_cpu_on_import_error(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert resolve_device(None) == "cpu"
