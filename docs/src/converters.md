# Module: `src.converters`

- File: `src/converters.py`
- Canonical equivalent: `src.od_training.dataset.convert`

## Purpose

Provides dataset format conversion and validation helpers:

- YOLO labels -> COCO JSON
- COCO JSON -> YOLO labels
- basic image/annotation integrity validation

## Types

### `ValidationReport`

Named tuple returned by `validate_dataset`:

- `ok: bool`
- `errors: list[tuple[str, str]]`
- `warnings: list[tuple[str, str]]`

## Functions

### `convert_yolo_to_coco(yolo_dir: str, image_dir: str, output_json: str)`

Uses `globox.AnnotationSet.from_yolo_v5` and writes COCO JSON via `save_coco`.

- Raises: conversion errors from `globox`/filesystem.

### `convert_coco_to_yolo(coco_json_dir: str, output_dir: str, use_segments: bool = False)`

Converts COCO annotations to YOLO labels.

- Primary backend: `ultralytics.data.converter.convert_coco`
- Fallback backend: `globox.AnnotationSet.from_coco(...).save_yolo_v5(...)`
- Creates output directory if missing.

### `validate_dataset(image_dir: str, label_dir: str, format: str = "yolo", class_names: list = None)`

Validates image files and YOLO label files.

Checks include:

- image readability/integrity
- matching label existence
- label column count (5 or 6)
- bbox value range `[0, 1]`
- class id range when `class_names` is provided

Returns `ValidationReport`.

### `_validate_yolo_label(label_path, class_names, errors, warnings)`

Internal helper used by `validate_dataset` to parse and validate one YOLO label file.

## Notes

Prefer importing from `src.od_training.dataset.convert` for new code paths.
