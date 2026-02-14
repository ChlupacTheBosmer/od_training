# Module: `src.od_training.utility.roboflow_download`

- File: `src/od_training/utility/roboflow_download.py`
- CLI command: `odt utility download-roboflow`

## Purpose

Downloads datasets from Roboflow, optionally unzips, and optionally imports into FiftyOne with split tagging.

## Functions

### `prompt_with_default(prompt: str, default: Optional[str] = None, required: bool = True) -> str`

Interactive prompt utility with default support and required-field loop.

### `get_interactive_params(args, config_path: Path | None = None)`

Interactive parameter completion workflow using config/env defaults where available.

### `download_dataset(api_key: str, workspace: str, project: str, version: int, format_type: str, download_dir: str) -> str`

Downloads a dataset version from Roboflow into `download_dir` and returns dataset location path.

### `unzip_dataset(download_dir: str) -> Optional[str]`

Unzips first `*.zip` archive found in `download_dir`.

Returns zip path when extraction happened, else `None`.

### `import_to_fiftyone(dataset_path: str, format_type: str, dataset_name: str, train_tag: str, val_tag: str, test_tag: str) -> bool`

Imports downloaded dataset into FiftyOne.

Behavior:

- supports YOLOv5/YOLOv8/YOLOv11 and COCO import paths
- applies split tags for YOLO formats based on source filepaths
- handles existing dataset overwrite prompt

Returns `True` on successful import, `False` otherwise.

### `main(argv=None)`

Top-level CLI orchestration:

1. parse args
2. optional interactive completion
3. resolve config/env defaults (supports `--config`)
4. download dataset
5. optional unzip and zip cleanup
6. optional FiftyOne import

Error handling:

- exits with code `1` for missing required parameters or runtime failures
- exits with code `0` on success
