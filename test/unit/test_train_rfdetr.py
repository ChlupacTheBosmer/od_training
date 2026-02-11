import sys
import types

import pytest


pytestmark = pytest.mark.unit


def _build_fake_rfdetr_module():
    mod = types.ModuleType("rfdetr")

    def make_cls(name):
        class _Cls:
            def __init__(self, *args, **kwargs):
                self._name = name

            def train(self, **kwargs):
                return kwargs

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
def train_rfdetr_module(load_script_module, monkeypatch):
    fake_rfdetr = _build_fake_rfdetr_module()
    fake_clearml = types.ModuleType("clearml")
    fake_clearml.Task = types.SimpleNamespace(init=lambda **_: object())

    monkeypatch.setitem(sys.modules, "rfdetr", fake_rfdetr)
    monkeypatch.setitem(sys.modules, "clearml", fake_clearml)

    return load_script_module("train_rfdetr.py")


def test_train_rfdetr_raises_for_missing_dataset(train_rfdetr_module):
    mod = train_rfdetr_module

    with pytest.raises(FileNotFoundError):
        mod.train_rfdetr(
            dataset_dir="/definitely/missing/path",
            model_type="rfdetr_nano",
            epochs=1,
            batch_size=1,
            lr=1e-4,
            project_name="proj",
            exp_name="exp",
            device="cpu",
        )


def test_train_rfdetr_defaults_to_medium_on_unknown_model(train_rfdetr_module, monkeypatch, tmp_path):
    mod = train_rfdetr_module
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    calls = {}

    class MediumModel:
        def __init__(self):
            calls["init"] = True

        def train(self, **kwargs):
            calls["train_args"] = kwargs

    monkeypatch.setattr(mod, "MODEL_MAP", {})
    monkeypatch.setattr(mod, "RFDETRMedium", MediumModel)

    mod.train_rfdetr(
        dataset_dir=str(dataset_dir),
        model_type="unknown-model",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        project_name="proj",
        exp_name="exp",
        device="cpu",
        tensorboard=False,
        wandb=False,
    )

    assert calls["init"] is True
    assert calls["train_args"]["dataset_dir"] == str(dataset_dir)
    assert calls["train_args"]["batch_size"] == 2


def test_train_rfdetr_tolerates_checkpoint_best_missing_when_other_checkpoint_exists(
    train_rfdetr_module, monkeypatch, tmp_path
):
    mod = train_rfdetr_module
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "checkpoint.pth").write_bytes(b"ok")

    class ModelWithCheckpointWarning:
        def train(self, **kwargs):
            raise FileNotFoundError("checkpoint_best_regular.pth not found")

    monkeypatch.setattr(mod, "MODEL_MAP", {"rfdetr_nano": lambda: ModelWithCheckpointWarning()})

    # Should not raise because output_dir already contains a .pth file
    mod.train_rfdetr(
        dataset_dir=str(dataset_dir),
        model_type="rfdetr_nano",
        epochs=1,
        batch_size=1,
        lr=1e-4,
        project_name="proj",
        exp_name="exp",
        device="cpu",
        output_dir=str(output_dir),
        tensorboard=False,
        wandb=False,
    )


def test_train_rfdetr_reraises_when_no_checkpoint_exists(train_rfdetr_module, monkeypatch, tmp_path):
    mod = train_rfdetr_module
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    class ModelWithCheckpointWarning:
        def train(self, **kwargs):
            raise FileNotFoundError("checkpoint_best_regular.pth not found")

    monkeypatch.setattr(mod, "MODEL_MAP", {"rfdetr_nano": lambda: ModelWithCheckpointWarning()})

    with pytest.raises(FileNotFoundError):
        mod.train_rfdetr(
            dataset_dir=str(dataset_dir),
            model_type="rfdetr_nano",
            epochs=1,
            batch_size=1,
            lr=1e-4,
            project_name="proj",
            exp_name="exp",
            device="cpu",
            output_dir=str(output_dir),
            tensorboard=False,
            wandb=False,
        )
