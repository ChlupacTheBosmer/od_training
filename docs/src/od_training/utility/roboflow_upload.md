# Module: `src.od_training.utility.roboflow_upload`

- File: `src/od_training/utility/roboflow_upload.py`
- CLI command: `odt utility upload-weights`

## Purpose

Uploads trained model weights to Roboflow Deploy.

## Functions

### `upload_weights(api_key: str, workspace: str, project_name: str, version_num: int, weights_path: str, model_type: str = "yolov8")`

Executes Roboflow deployment API call.

Behavior:

- validates weights file exists and is non-empty
- lazy-imports `roboflow` package
- resolves project/version and calls `version.deploy(...)`

Raises:

- `FileNotFoundError` for missing weights
- `ValueError` for empty weights file
- `ImportError` when `roboflow` package is unavailable
- deployment exceptions from Roboflow API

### `build_parser() -> argparse.ArgumentParser`

Defines CLI args:

- `--config`, `--api-key`, `--workspace`, `--project`, `--version`, `--weights`, `--type`

### `main(argv=None)`

Resolves credentials and defaults via runtime config helpers (including optional `--config` override), validates required workspace/project values, then calls `upload_weights`.

Returns `0` on success.
