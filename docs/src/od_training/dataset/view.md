# Module: `src.od_training.dataset.view`

- File: `src/od_training/dataset/view.py`
- CLI command: `odt dataset view`

## Purpose

Loads an existing FiftyOne dataset by name, or imports one from disk when missing.

## Functions

### `check_dataset_exists(name)`

Returns `True` if `name` is present in `fo.list_datasets()`.

### `import_dataset(name: str, import_dir: str)`

Imports YOLO dataset into FiftyOne.

Behavior:

- Accepts directory or YAML path.
- If a file path is provided, infers dataset directory and passes `yaml_path`.
- Uses `fo.Dataset.from_dir(..., dataset_type=fo.types.YOLOv5Dataset)`.

### `build_parser() -> argparse.ArgumentParser`

Defines CLI args:

- positional `name`
- optional `--import-dir`

### `main(argv=None)`

Workflow:

1. parse args
2. load existing dataset or import when `--import-dir` is provided
3. launch FiftyOne app session and block via `session.wait()`

Returns:

- `0` on success
- `1` when dataset is missing and no import path is provided
