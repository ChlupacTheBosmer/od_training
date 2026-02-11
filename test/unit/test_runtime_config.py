import json

import pytest

from src import runtime_config


pytestmark = pytest.mark.unit


def test_ensure_local_config_creates_template(tmp_path):
    config_path = tmp_path / "config" / "local_config.json"

    path, created = runtime_config.ensure_local_config(config_path)

    assert path == config_path
    assert created is True
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert "roboflow" in data
    assert "api_key" in data["roboflow"]


def test_ensure_local_config_no_overwrite(tmp_path):
    config_path = tmp_path / "config" / "local_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('{"roboflow": {"api_key": "abc"}}\n', encoding="utf-8")

    _, created = runtime_config.ensure_local_config(config_path)
    assert created is False


def test_load_local_config_invalid_json(tmp_path):
    config_path = tmp_path / "config" / "local_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError):
        runtime_config.load_local_config(config_path)


def test_get_roboflow_api_key_uses_explicit_value():
    assert runtime_config.get_roboflow_api_key("cli-key") == "cli-key"


def test_get_roboflow_api_key_reads_from_config(monkeypatch):
    monkeypatch.setattr(
        runtime_config,
        "load_local_config",
        lambda config_path=None: {"roboflow": {"api_key": "abc123"}},
    )

    assert runtime_config.get_roboflow_api_key(None) == "abc123"


def test_get_roboflow_api_key_raises_for_placeholder(monkeypatch):
    monkeypatch.setattr(
        runtime_config,
        "load_local_config",
        lambda config_path=None: {"roboflow": {"api_key": "<PASTE_ROBOFLOW_API_KEY_HERE>"}},
    )

    with pytest.raises(ValueError):
        runtime_config.get_roboflow_api_key(None)


def test_get_roboflow_default(monkeypatch):
    monkeypatch.setattr(
        runtime_config,
        "load_local_config",
        lambda config_path=None: {
            "roboflow": {
                "workspace": "ws",
                "project": "proj",
            }
        },
    )

    assert runtime_config.get_roboflow_default("workspace") == "ws"
    assert runtime_config.get_roboflow_default("project") == "proj"


def test_get_roboflow_default_returns_none_for_placeholder(monkeypatch):
    monkeypatch.setattr(
        runtime_config,
        "load_local_config",
        lambda config_path=None: {"roboflow": {"workspace": "<OPTIONAL_DEFAULT_WORKSPACE_ID>"}},
    )

    assert runtime_config.get_roboflow_default("workspace") is None
