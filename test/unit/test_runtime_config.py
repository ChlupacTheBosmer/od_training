import json

import pytest

from od_training.utility import runtime_config


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
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)

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
    monkeypatch.delenv("ROBOFLOW_WORKSPACE", raising=False)

    assert runtime_config.get_roboflow_default("workspace") is None


def test_resolve_default_local_config_path_env_override(monkeypatch, tmp_path):
    path = tmp_path / "custom.json"
    monkeypatch.setenv("ODT_CONFIG_PATH", str(path))
    assert runtime_config.resolve_default_local_config_path() == path


def test_get_config_path_explicit_argument(tmp_path):
    path = tmp_path / "manual.json"
    assert runtime_config.get_config_path(path) == path


def test_parse_local_config_returns_typed_model():
    cfg = runtime_config.parse_local_config(
        {
            "_instructions": {"note": "x"},
            "roboflow": {"api_key": "abc", "workspace": "ws", "project": "proj"},
            "data_dir": "data_local",
        }
    )

    assert cfg.roboflow.api_key == "abc"
    assert cfg.roboflow.workspace == "ws"
    assert cfg.roboflow.project == "proj"
    assert cfg.data_dir == "data_local"


def test_parse_local_config_rejects_invalid_schema():
    with pytest.raises(ValueError):
        runtime_config.parse_local_config({"roboflow": {"api_key": "abc"}, "data_dir": []})
