# Module: `src.od_training.dataset.manager`

- File: `src/od_training/dataset/manager.py`
- CLI command: `odt dataset manage`

## Purpose

Primary dataset orchestration module for import/split/augment/export tasks.

## Functions

### `get_augmentation_pipeline(width=640, height=640)`

Builds Albumentations transform used by `augment_samples`.

### `load_or_create_dataset(dataset_dir, name, split_ratios=None, train_tag="train", val_tag="val", test_tag="test")`

Loads existing FiftyOne dataset or imports dataset from disk.

Behavior:

- Accepts directory or YAML path.
- Auto-detects `data.yaml` when needed.
- Optional split operation with custom split tag names.
- Split is skipped if any of provided split tags already exist.

### `augment_samples(dataset, filter_tags=None, new_dataset_name=None, output_dir=None, num_aug=1, label_field="ground_truth")`

Creates augmented samples and appends them either to source dataset or a target dataset.

Behavior:

- Supports tag filtering.
- Writes augmented images to `output_dir` or `data/augmented/<dataset_name>`.
- Preserves original tags and appends `augmented` tag.
- Reads and writes detections through `label_field`.
- Returns destination dataset instance.

### `export_pipeline(dataset, export_dir, classes=None, label_field="ground_truth", classes_field=None, train_tag="train", val_tag="val", test_tag="test", export_tags=None, copy_images=False, include_confidence=False)`

Exports dataset in hybrid format:

- YOLOv5 format (`images/`, `labels/`, `dataset.yaml`)
- COCO JSON (`_annotations_<split>.coco.json`)

Modes:

- symlink media mode (default)
- copied media mode (`copy_images=True`)

Supports split tag remapping and all-tag filtering (`export_tags`).
The exported detection source is configurable via `label_field`.
When `classes` is omitted, class discovery can be overridden via
`classes_field` and invalid paths raise clear errors.

### `_fix_coco_filenames(json_path)`

Normalizes COCO `file_name` to basename only when safe.

Behavior:

- detects duplicate basenames
- preserves full paths when collisions are detected

### `build_parser() -> argparse.ArgumentParser`

Defines CLI flags for all dataset operations:

- load/split: `--dataset-dir`, `--name`, `--split`
- augment: `--augment`, `--augment-tags`, `--output-dataset`, `--output-dir`
- export: `--export-dir`, `--export-tags`, `--copy-images`, `--include-confidence`
- label/class selection: `--label-field`, `--classes-field`
- view: `--view`
- split tags: `--train-tag`, `--val-tag`, `--test-tag`

### `main(argv=None)`

Top-level workflow:

1. load/create dataset
2. optionally augment
3. optionally export
4. optionally launch FiftyOne app

Returns `0` on successful execution.

Validation behavior:

- Export fails early with a clear `ValueError` if `label_field` is missing.
- Export fails early with a clear `ValueError` if class discovery path is invalid.
