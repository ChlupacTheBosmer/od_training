# Module: `src.od_training.train.rfdetr`

- File: `src/od_training/train/rfdetr.py`
- CLI command: `odt train rfdetr`

## Purpose

Runs RF-DETR training on local COCO-style datasets and applies checkpoint-related resilience for short runs.

## Globals

### `MODEL_MAP`

Maps model keys to RF-DETR classes, including aliases:

- `rfdetr_nano`, `rfdetr_small`, `rfdetr_medium`, `rfdetr_large`, `rfdetr_xlarge`, `rfdetr_2xlarge`
- `rf-detr-resnet50` -> `RFDETRMedium`
- `rf-detr-base` -> `RFDETRBase`

## Functions

### `train_rfdetr(dataset_dir: str, model_type: str, epochs: int, batch_size: int, lr: float, project_name: str, exp_name: str, device: str = None, validate_data: bool = False, fail_on_validation_warnings: bool = False, **kwargs)`

Constructs RF-DETR model instance and runs `model.train(**train_args)`.

Behavior:

- Optional preflight validates RF-DETR split contract (`train/valid/test` + `_annotations.coco.json`).
- Initializes ClearML task when available.
- Resolves model class from `MODEL_MAP` with substring fallback.
- Falls back to `RFDETRMedium` when model key is unknown.
- Raises `FileNotFoundError` if dataset directory is missing.
- Default `train_args` include `tensorboard=True`, `wandb=True`.
- Handles `checkpoint_best_regular.pth` post-train copy issue by verifying whether any `.pth` checkpoint exists before failing.

### `build_parser() -> argparse.ArgumentParser`

Defines CLI arguments:

- `--dataset`, `--model`, `--epochs`, `--batch`, `--lr`, `--project`, `--name`, `--device`
- `--no-validate-data`, `--fail-on-validation-warnings`

### `main(argv=None) -> int`

Parses known args, parses unknown passthrough args, invokes `train_rfdetr`, returns `0` on success.
