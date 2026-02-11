# Module: `src.od_training.dataset.augment_preview`

- File: `src/od_training/dataset/augment_preview.py`
- CLI command: `odt dataset augment-preview`

## Purpose

Provides an interactive augmentation preview workflow for YOLO datasets in FiftyOne.

## Functions

### `define_pipeline(width=640, height=640)`

Returns Albumentations pipeline used for preview:

- random crop
- horizontal flip
- brightness/contrast
- grayscale conversion

### `augment_and_view(dataset_dir: str, name: str = "my_dataset", yaml_path: str = None)`

Loads/imports dataset, applies one preview augmentation, draws boxes, and writes preview image.

Key behavior:

- Creates `runs/augmentation_test/aug_sample.jpg` when preview succeeds.
- Prints operational instructions for continuing augmentation experimentation in FiftyOne UI.
- Does not auto-launch app session by default (launch code is commented out).

### `build_parser() -> argparse.ArgumentParser`

Defines CLI args:

- `--dataset` (required)
- `--name` (optional)
- `--yaml` (optional)

### `main(argv=None)`

Parses args, executes `augment_and_view`, returns `0`.
