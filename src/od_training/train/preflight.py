"""Preflight dataset validation helpers for training commands.

These checks are intentionally lightweight and deterministic so they can fail
fast before long-running training jobs start.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..dataset.convert import validate_dataset

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _as_class_names(raw_names: Any) -> list[str] | None:
    """Normalize `names` from YOLO YAML into ordered class names."""
    if isinstance(raw_names, list):
        return [str(v) for v in raw_names]
    if isinstance(raw_names, dict):
        items: list[tuple[int, str]] = []
        for key, value in raw_names.items():
            try:
                idx = int(key)
            except Exception:
                continue
            items.append((idx, str(value)))
        if not items:
            return None
        return [name for _, name in sorted(items, key=lambda x: x[0])]
    return None


def _resolve_split_path(
    *,
    split_value: Any,
    yaml_dir: Path,
    data_root: Path,
) -> Path | None:
    """Resolve a split path from YOLO YAML content."""
    if not isinstance(split_value, str):
        return None
    split = Path(split_value)
    if split.is_absolute():
        return split
    # Prefer `path:` root semantics from YOLO YAML, then YAML dir fallback.
    candidate = data_root / split
    if candidate.exists():
        return candidate
    return (yaml_dir / split).resolve()


def _resolve_labels_dir(images_dir: Path) -> Path:
    """Infer labels dir from images dir using common YOLO directory conventions."""
    if images_dir.name == "images":
        return images_dir.parent / "labels"
    if images_dir.parent.name == "images":
        return images_dir.parent.parent / "labels" / images_dir.name
    return images_dir.parent / "labels"


def validate_yolo_training_inputs(
    *,
    data_yaml: str,
    fail_on_warnings: bool = False,
) -> dict[str, Any]:
    """Validate YOLO training dataset inputs from a data YAML.

    Returns a machine-readable summary and raises `ValueError` on blocking
    validation failures.
    """
    yaml_path = Path(data_yaml).expanduser().resolve()
    if not yaml_path.exists():
        raise ValueError(f"YOLO data YAML not found: '{yaml_path}'")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YOLO data YAML must be a mapping object: '{yaml_path}'")

    yaml_dir = yaml_path.parent
    data_root_raw = data.get("path")
    if isinstance(data_root_raw, str) and data_root_raw.strip():
        data_root = Path(data_root_raw)
        if not data_root.is_absolute():
            data_root = (yaml_dir / data_root).resolve()
    else:
        data_root = yaml_dir

    class_names = _as_class_names(data.get("names"))
    split_key_map = {"train": "train", "val": "val", "test": "test"}

    errors: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    checked_splits: list[str] = []

    for split_name, yaml_key in split_key_map.items():
        split_value = data.get(yaml_key)
        if split_value is None:
            continue

        split_path = _resolve_split_path(
            split_value=split_value,
            yaml_dir=yaml_dir,
            data_root=data_root,
        )
        if split_path is None:
            warnings.append(
                (
                    split_name,
                    f"Unsupported split path type for '{yaml_key}': {type(split_value).__name__}",
                )
            )
            continue

        # Common YOLO variant: split references a .txt list; skip deep validation.
        if split_path.suffix.lower() == ".txt":
            warnings.append(
                (
                    split_name,
                    f"Split '{yaml_key}' uses txt file list ('{split_path}'); skipping deep image/label validation.",
                )
            )
            continue

        if not split_path.exists():
            errors.append((split_name, f"Split path does not exist: '{split_path}'"))
            continue
        if not split_path.is_dir():
            errors.append((split_name, f"Split path is not a directory: '{split_path}'"))
            continue

        labels_dir = _resolve_labels_dir(split_path)
        if not labels_dir.exists() or not labels_dir.is_dir():
            errors.append(
                (
                    split_name,
                    f"Inferred labels dir does not exist: '{labels_dir}' (from images dir '{split_path}')",
                )
            )
            continue

        report = validate_dataset(
            image_dir=str(split_path),
            label_dir=str(labels_dir),
            format="yolo",
            class_names=class_names,
        )
        checked_splits.append(split_name)
        errors.extend((split_name, msg) for _, msg in report.errors)
        warnings.extend((split_name, msg) for _, msg in report.warnings)

    if not checked_splits:
        warnings.append(
            (
                "preflight",
                "No train/val/test directory splits were validated from YAML configuration.",
            )
        )

    if errors:
        first = "; ".join(f"[{k}] {v}" for k, v in errors[:5])
        raise ValueError(
            f"YOLO preflight validation failed with {len(errors)} error(s): {first}"
        )
    if fail_on_warnings and warnings:
        first = "; ".join(f"[{k}] {v}" for k, v in warnings[:5])
        raise ValueError(
            f"YOLO preflight validation failed due to {len(warnings)} warning(s): {first}"
        )

    return {
        "ok": True,
        "checked_splits": checked_splits,
        "errors": errors,
        "warnings": warnings,
    }


def _count_images(path: Path) -> int:
    """Count supported image files directly under `path`."""
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)


def validate_rfdetr_training_inputs(
    *,
    dataset_dir: str,
    fail_on_warnings: bool = False,
) -> dict[str, Any]:
    """Validate RF-DETR dataset directory contract before training."""
    root = Path(dataset_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"RF-DETR dataset directory not found: '{root}'")

    errors: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    split_details: dict[str, dict[str, Any]] = {}

    for split in ("train", "valid", "test"):
        split_dir = root / split
        if not split_dir.exists() or not split_dir.is_dir():
            errors.append((split, f"Missing split directory: '{split_dir}'"))
            continue

        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            errors.append((split, f"Missing COCO annotation file: '{ann_path}'"))
            continue

        image_count = _count_images(split_dir)
        if image_count == 0:
            warnings.append((split, f"No images found in split directory: '{split_dir}'"))

        try:
            with ann_path.open("r", encoding="utf-8") as f:
                ann = json.load(f)
        except Exception as exc:
            errors.append((split, f"Invalid COCO JSON '{ann_path}': {exc}"))
            continue

        for key in ("images", "annotations", "categories"):
            if key not in ann:
                errors.append((split, f"COCO file missing key '{key}': '{ann_path}'"))

        categories = ann.get("categories", [])
        if isinstance(categories, list) and len(categories) == 0:
            warnings.append((split, f"COCO categories list is empty: '{ann_path}'"))

        split_details[split] = {
            "split_dir": str(split_dir),
            "annotation_file": str(ann_path),
            "image_count": image_count,
        }

    if errors:
        first = "; ".join(f"[{k}] {v}" for k, v in errors[:5])
        raise ValueError(
            f"RF-DETR preflight validation failed with {len(errors)} error(s): {first}"
        )
    if fail_on_warnings and warnings:
        first = "; ".join(f"[{k}] {v}" for k, v in warnings[:5])
        raise ValueError(
            f"RF-DETR preflight validation failed due to {len(warnings)} warning(s): {first}"
        )

    return {
        "ok": True,
        "dataset_dir": str(root),
        "splits": split_details,
        "errors": errors,
        "warnings": warnings,
    }

