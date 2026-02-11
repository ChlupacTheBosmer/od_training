# Module: `src.od_training.dataset.convert_cli`

- File: `src/od_training/dataset/convert_cli.py`
- CLI command: `odt dataset convert`

## Purpose

Thin CLI adapter that exposes conversion functions from `src.od_training.dataset.convert`.

## Functions

### `main(argv=None)`

CLI behavior:

- Positional `mode`: `yolo2coco` or `coco2yolo`
- Required args: `--input`, `--output`
- Additional required arg for `yolo2coco`: `--images`

Dispatch:

- `yolo2coco` -> `convert_yolo_to_coco`
- `coco2yolo` -> `convert_coco_to_yolo`

Exits with status `1` when `--images` is missing for YOLO->COCO conversion.
