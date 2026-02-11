import pytest

from src.od_training.cli import main as cli_main


pytestmark = pytest.mark.unit


def test_odt_cli_unknown_command_returns_2(capsys):
    rc = cli_main(["train", "unknown"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "Unknown command: train unknown" in out


def test_odt_cli_dispatch_forwards_args(monkeypatch):
    captured = {}

    def fake_entry(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setitem(cli_main.__globals__["DISPATCH"], ("train", "yolo"), fake_entry)
    rc = cli_main(["train", "yolo", "--", "--epochs", "1", "--data", "data.yaml"])

    assert rc == 0
    assert captured["argv"] == ["--epochs", "1", "--data", "data.yaml"]
