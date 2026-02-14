import json
import types
from pathlib import Path

import pytest

from od_training.dataset import manager as dataset_manager_mod

pytestmark = pytest.mark.unit


class _FakeView:
    def __init__(self, samples, recorder):
        self.samples = samples
        self.recorder = recorder

    def match_tags(self, tag):
        filtered = [s for s in self.samples if tag in s.get("tags", [])]
        return _FakeView(filtered, self.recorder)

    def __len__(self):
        return len(self.samples)

    def export(self, **kwargs):
        payload = {"count": len(self.samples), **kwargs}
        self.recorder.append(payload)

        # Mimic COCO export artifact so _fix_coco_filenames path executes
        labels_path = kwargs.get("labels_path")
        if labels_path:
            p = Path(labels_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 1,
                                "file_name": "/tmp/image.jpg",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )


class _FakeDataset(_FakeView):
    def __init__(self, samples, recorder):
        super().__init__(samples, recorder)
        self.name = "fake_ds"


class _SplitDataset:
    def __init__(self, counts):
        self._counts = counts
        self.saved = False

    def count_sample_tags(self):
        return self._counts

    def save(self):
        self.saved = True


class _EmptyMatchDataset:
    def __init__(self):
        self.name = "empty"

    def match_tags(self, _):
        return []

    def __len__(self):
        return 0


@pytest.fixture
def dataset_manager_module():
    return dataset_manager_mod


def test_fix_coco_filenames_unique(dataset_manager_module, tmp_path):
    mod = dataset_manager_module

    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [
                    {"file_name": "/a/b/img1.jpg"},
                    {"file_name": "/c/d/img2.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )

    mod._fix_coco_filenames(str(p))
    data = json.loads(p.read_text(encoding="utf-8"))

    assert data["images"][0]["file_name"] == "img1.jpg"
    assert data["images"][1]["file_name"] == "img2.jpg"


def test_fix_coco_filenames_duplicates_preserve_full_paths(dataset_manager_module, tmp_path):
    mod = dataset_manager_module

    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [
                    {"file_name": "/a/b/dup.jpg"},
                    {"file_name": "/x/y/dup.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )

    mod._fix_coco_filenames(str(p))
    data = json.loads(p.read_text(encoding="utf-8"))

    assert data["images"][0]["file_name"] == "/a/b/dup.jpg"
    assert data["images"][1]["file_name"] == "/x/y/dup.jpg"


def test_export_pipeline_symlink_mode(dataset_manager_module, monkeypatch, tmp_path):
    mod = dataset_manager_module

    recorder = []
    samples = [{"tags": ["train", "augmented"]}]
    dataset = _FakeDataset(samples, recorder)

    monkeypatch.setattr(
        mod.fo,
        "types",
        types.SimpleNamespace(YOLOv5Dataset="YOLO", COCODetectionDataset="COCO"),
        raising=False,
    )

    mod.export_pipeline(
        dataset,
        str(tmp_path / "export"),
        classes=["object"],
        copy_images=False,
        export_tags=["augmented"],
    )

    yolo_calls = [c for c in recorder if c.get("dataset_type") == "YOLO"]
    coco_calls = [c for c in recorder if c.get("dataset_type") == "COCO"]

    assert len(yolo_calls) == 1
    assert len(coco_calls) == 1
    assert yolo_calls[0]["export_media"] == "symlink"
    assert yolo_calls[0]["count"] == 1


def test_export_pipeline_copy_mode(dataset_manager_module, monkeypatch, tmp_path):
    mod = dataset_manager_module

    recorder = []
    samples = [{"tags": ["train"]}]
    dataset = _FakeDataset(samples, recorder)

    monkeypatch.setattr(
        mod.fo,
        "types",
        types.SimpleNamespace(YOLOv5Dataset="YOLO", COCODetectionDataset="COCO"),
        raising=False,
    )

    mod.export_pipeline(dataset, str(tmp_path / "export"), classes=["object"], copy_images=True)

    yolo_calls = [c for c in recorder if c.get("dataset_type") == "YOLO"]
    assert yolo_calls[0]["export_media"] is True


def test_export_pipeline_respects_label_field(dataset_manager_module, monkeypatch, tmp_path):
    mod = dataset_manager_module

    recorder = []
    samples = [{"tags": ["train"]}]
    dataset = _FakeDataset(samples, recorder)

    monkeypatch.setattr(
        mod.fo,
        "types",
        types.SimpleNamespace(YOLOv5Dataset="YOLO", COCODetectionDataset="COCO"),
        raising=False,
    )

    mod.export_pipeline(
        dataset,
        str(tmp_path / "export"),
        classes=["object"],
        label_field="ls_corrections",
    )

    yolo_calls = [c for c in recorder if c.get("dataset_type") == "YOLO"]
    coco_calls = [c for c in recorder if c.get("dataset_type") == "COCO"]
    assert yolo_calls[0]["label_field"] == "ls_corrections"
    assert coco_calls[0]["label_field"] == "ls_corrections"


def test_augment_samples_returns_early_when_no_samples(dataset_manager_module):
    mod = dataset_manager_module

    class EmptyDataset:
        name = "demo"

        def match_tags(self, _):
            return []

    dataset = EmptyDataset()
    result = mod.augment_samples(dataset, filter_tags=["train"])
    assert result is dataset


def test_load_or_create_dataset_uses_existing_dataset(dataset_manager_module, monkeypatch):
    mod = dataset_manager_module

    sentinel = object()
    monkeypatch.setattr(mod.fo, "list_datasets", lambda: ["my_ds"])
    monkeypatch.setattr(mod.fo, "load_dataset", lambda _: sentinel)

    result = mod.load_or_create_dataset("ignored", "my_ds")
    assert result is sentinel


def test_load_or_create_dataset_triggers_split_when_missing_tags(dataset_manager_module, monkeypatch):
    mod = dataset_manager_module

    ds = _SplitDataset(counts={})
    calls = {}

    monkeypatch.setattr(mod.fo, "list_datasets", lambda: ["my_ds"])
    monkeypatch.setattr(mod.fo, "load_dataset", lambda _: ds)

    def fake_random_split(dataset, mapping):
        calls["dataset"] = dataset
        calls["mapping"] = mapping

    monkeypatch.setattr(mod.four, "random_split", fake_random_split)

    result = mod.load_or_create_dataset(
        "ignored",
        "my_ds",
        split_ratios={"train": 0.7, "val": 0.2, "test": 0.1},
        train_tag="tr",
        val_tag="va",
        test_tag="te",
    )

    assert result is ds
    assert calls["dataset"] is ds
    assert calls["mapping"] == {"tr": 0.7, "va": 0.2, "te": 0.1}
    assert ds.saved is True


def test_main_uses_classes_field_override(dataset_manager_module, monkeypatch, tmp_path):
    mod = dataset_manager_module
    calls = {}

    class _Dataset:
        name = "demo"
        default_classes = None

        def distinct(self, field):
            calls["distinct_field"] = field
            return ["rodent"]

    monkeypatch.setattr(mod, "load_or_create_dataset", lambda *args, **kwargs: _Dataset())

    def fake_export_pipeline(dataset, export_dir, **kwargs):
        calls["label_field"] = kwargs["label_field"]
        calls["classes"] = kwargs["classes"]
        calls["export_dir"] = export_dir

    monkeypatch.setattr(mod, "export_pipeline", fake_export_pipeline)

    rc = mod.main(
        [
            "--name",
            "demo",
            "--export-dir",
            str(tmp_path / "out"),
            "--label-field",
            "ls_corrections",
            "--classes-field",
            "ls_corrections.detections.label",
        ]
    )

    assert rc == 0
    assert calls["distinct_field"] == "ls_corrections.detections.label"
    assert calls["label_field"] == "ls_corrections"
    assert calls["classes"] == ["rodent"]


def test_main_export_raises_clear_error_for_missing_label_field(dataset_manager_module, monkeypatch, tmp_path):
    mod = dataset_manager_module

    class _Dataset:
        name = "demo"
        default_classes = None

        def has_sample_field(self, field):
            return field == "ground_truth"

        def get_field_schema(self):
            return {"ground_truth": object(), "predictions": object()}

        def distinct(self, field):
            return ["rodent"]

    monkeypatch.setattr(mod, "load_or_create_dataset", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(
        mod,
        "export_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("export should not be called")),
    )

    with pytest.raises(ValueError, match="Label field 'ls_corrections' was not found"):
        mod.main(
            [
                "--name",
                "demo",
                "--export-dir",
                str(tmp_path / "out"),
                "--label-field",
                "ls_corrections",
            ]
        )


def test_main_export_raises_clear_error_for_invalid_classes_field(dataset_manager_module, monkeypatch, tmp_path):
    mod = dataset_manager_module

    class _Dataset:
        name = "demo"
        default_classes = None

        def has_sample_field(self, field):
            return field == "ground_truth"

        def distinct(self, field):
            raise RuntimeError("bad field")

    monkeypatch.setattr(mod, "load_or_create_dataset", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(
        mod,
        "export_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("export should not be called")),
    )

    with pytest.raises(ValueError, match="Unable to resolve export classes"):
        mod.main(
            [
                "--name",
                "demo",
                "--export-dir",
                str(tmp_path / "out"),
                "--label-field",
                "ground_truth",
                "--classes-field",
                "ls_corrections.detections.label",
            ]
        )
