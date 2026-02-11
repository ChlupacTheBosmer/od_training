# Module: `src.od_training.infer.runner`

- File: `src/od_training/infer/runner.py`
- CLI command: `odt infer run`

## Purpose

Runs inference for YOLO and RF-DETR models across image files, image directories, videos, webcam, and RTSP sources.

## Globals

- `RFDETR_AVAILABLE`: set at import time based on `rfdetr` import success.
- `RFDETR_MAP`: architecture key -> class mapping for RF-DETR models.

## Functions

### `load_rfdetr_model(model_name: str, rfdetr_arch: str = None)`

Loads RF-DETR model using either:

- known architecture key from `RFDETR_MAP`, or
- custom weights path with inferred/explicit architecture.

Raises:

- `ImportError` when RF-DETR package is unavailable.
- `FileNotFoundError` for unresolved custom weights path.

### `run_inference(source: str, model_name: str, model_type: str, rfdetr_arch: str = None, conf_threshold: float = 0.3, iou_threshold: float = 0.5, show: bool = False, save_dir: str = None)`

Performs end-to-end inference and annotation.

Behavior:

- `model_type="yolo"`: calls `YOLO(...).predict(...)`.
- `model_type="rfdetr"`: calls `load_rfdetr_model(...).predict(...)`.
- Supports sources:
  - image file
  - directory of images (`.jpg/.jpeg/.png`)
  - video file
  - webcam (`0`)
  - RTSP URL
- Annotates detections with `supervision` annotators.
- Optional display via OpenCV window (`--show`).
- Optional writes to `save_dir`:
  - images as `<stem>_annotated.jpg`
  - video as `output_video.mp4`

Raises `ValueError` for unsupported model type or unreadable source.

### `build_parser() -> argparse.ArgumentParser`

Defines CLI arguments:

- `--source`, `--model`, `--type`, `--rfdetr-arch`, `--conf`, `--iou`, `--show`, `--save-dir`

### `main(argv=None) -> int`

Parses args, calls `run_inference`, and returns `0` on success.
