# Module: `src.od_training.train.yolo`

- File: `src/od_training/train/yolo.py`
- CLI command: `odt train yolo`

## Purpose

Runs YOLO training with optional ClearML tracking and post-training TensorRT export.

## Functions

### `train_yolo(model_name: str, data_yaml: str, project_name: str, exp_name: str, *, validate_data=False, fail_on_validation_warnings=False, **kwargs)`

Loads a YOLO model and calls `model.train(...)`.

Parameters:

- `model_name`: model weights/specifier (for example `yolo11n.pt`).
- `data_yaml`: path to dataset YAML.
- `project_name`: run/project directory or logical project name.
- `exp_name`: experiment name.
- `validate_data`: run preflight dataset validation before training.
- `fail_on_validation_warnings`: escalate preflight warnings to failures.
- `**kwargs`: forwarded to Ultralytics training API.

Behavior:

- Optional preflight validates split image/label structure from YOLO YAML.
- Attempts `Task.init(...)` for ClearML; continues if tracking init fails.
- If model name contains `yolo26` and optimizer is missing, sets `optimizer="MuSGD"`.
- If `device != "cpu"` and `epochs > 1`, attempts TensorRT export.

Returns Ultralytics training result object.

### `build_parser() -> argparse.ArgumentParser`

Defines CLI arguments:

- `--model`, `--data`, `--project`, `--name`, `--epochs`, `--batch`, `--imgsz`, `--device`
- `--no-validate-data`, `--fail-on-validation-warnings`

### `main(argv=None) -> int`

Parses known args, converts unknown args via `parse_unknown_args`, resolves device via `resolve_device`, invokes `train_yolo`, and returns `0` on success.
