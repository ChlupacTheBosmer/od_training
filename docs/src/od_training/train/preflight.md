# Module: `src.od_training.train.preflight`

- File: `src/od_training/train/preflight.py`

## Purpose

Preflight validation helpers used by training commands to catch dataset
contract issues before long-running jobs start.

## Functions

### `validate_yolo_training_inputs(data_yaml: str, fail_on_warnings: bool = False) -> dict[str, Any]`

Validates YOLO dataset structure from `data.yaml`:

- resolves split paths (`train`, `val`, `test`)
- infers corresponding labels directories
- runs `dataset.convert.validate_dataset(...)` per split

Fails on validation errors. Optionally fails on warnings when
`fail_on_warnings=True`.

### `validate_rfdetr_training_inputs(dataset_dir: str, fail_on_warnings: bool = False) -> dict[str, Any]`

Validates RF-DETR split contract:

- requires `train/`, `valid/`, `test/` directories
- requires `_annotations.coco.json` inside each split dir
- checks basic COCO keys (`images`, `annotations`, `categories`)

Fails on validation errors. Optionally fails on warnings when
`fail_on_warnings=True`.

