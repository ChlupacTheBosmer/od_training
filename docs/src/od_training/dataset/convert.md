# Module: `src.od_training.dataset.convert`

- File: `src/od_training/dataset/convert.py`

## Purpose

Implements format conversion and dataset validation primitives.

## Types

### `ValidationReport`

Named tuple used by `validate_dataset`:

- `ok`: `True` when no errors were found.
- `errors`: list of `(path/context, message)` entries.
- `warnings`: list of `(path/context, message)` entries.

## Functions

### `convert_yolo_to_coco(yolo_dir: str, image_dir: str, output_json: str)`

Converts YOLO labels to COCO JSON using `globox`.

### `convert_coco_to_yolo(coco_json_dir: str, output_dir: str, use_segments: bool = False)`

Converts COCO JSON to YOLO labels.

- Primary path: Ultralytics converter.
- Fallback path: `globox` when Ultralytics converter API is unavailable.

### `validate_dataset(image_dir: str, label_dir: str, format: str = "yolo", class_names: list = None)`

Validates image and label consistency.

Checks:

- image readability
- label existence
- column count (5/6)
- bbox range validation
- class-id range (when class names are provided)

Returns `ValidationReport`.

### `_validate_yolo_label(label_path, class_names, errors, warnings)`

Internal line-level YOLO label validation helper.
