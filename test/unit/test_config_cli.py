import json
from pathlib import Path

import pytest

from od_training.utility import config_cli


pytestmark = pytest.mark.unit


def test_config_init_creates_local_config(tmp_path, capsys):
    cfg_path = tmp_path / "cfg" / "local_config.json"

    rc = config_cli.main_init(["--config", str(cfg_path)])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert cfg_path.exists()
    assert payload["ok"] is True
    assert payload["path"] == str(cfg_path)


def test_config_show_masks_secret_fields(tmp_path, capsys):
    cfg_path = tmp_path / "cfg" / "local_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps(
            {
                "roboflow": {
                    "api_key": "rf_secret",
                    "workspace": "ws",
                    "project": "proj",
                },
                "data_dir": "data",
            }
        ),
        encoding="utf-8",
    )

    rc = config_cli.main_show(["--config", str(cfg_path), "--mask-secrets"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["roboflow"]["api_key"] == "***MASKED***"
    assert payload["roboflow"]["workspace"] == "ws"
    assert payload["roboflow"]["project"] == "proj"


def test_config_show_without_mask_keeps_raw_value(tmp_path, capsys):
    cfg_path = tmp_path / "cfg" / "local_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps(
            {
                "roboflow": {
                    "api_key": "rf_secret",
                },
            }
        ),
        encoding="utf-8",
    )

    rc = config_cli.main_show(["--config", str(cfg_path)])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["roboflow"]["api_key"] == "rf_secret"

