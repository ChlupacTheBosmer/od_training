# Development Status

**Last updated:** 2026-02-11
**Purpose:** Current, repo-local status of implemented functionality and open work.

## Implemented

### CLI And Structure
- Unified CLI entrypoint: `scripts/odt.py`
- Modular dispatch: `src/od_training/cli/main.py`
- Command groups live under:
  - `src/od_training/dataset/`
  - `src/od_training/train/`
  - `src/od_training/infer/`
  - `src/od_training/utility/`

### Dataset Pipeline
- YOLO import/split/augment/export pipeline implemented in `src/od_training/dataset/manager.py`
- Export defaults to zero-copy symlink mode (`export_media="symlink"`) with `--copy-images` for portable copies
- COCO sidecar exports generated per split (`_annotations_<split>.coco.json`)
- Optional confidence-preserving YOLO export via `--include-confidence`

### Training
- YOLO training wrapper: `src/od_training/train/yolo.py`
- RF-DETR training wrapper: `src/od_training/train/rfdetr.py`
- Shared device selection and unknown-arg parsing via:
  - `src/od_training/utility/device.py`
  - `src/od_training/utility/cli.py`

### Inference
- Unified inference runner: `src/od_training/infer/runner.py`
- Supports YOLO and RF-DETR, image/video/dir/webcam/RTSP inputs
- Output naming and save-dir behavior implemented

### Utilities
- Runtime config with env var fallback: `src/od_training/utility/runtime_config.py`
- Environment verification: `src/od_training/utility/verify_env.py`
- Roboflow download/upload tooling:
  - `src/od_training/utility/roboflow_download.py`
  - `src/od_training/utility/roboflow_upload.py`

## In Progress / Pending
- Typed config model (Pydantic `BaseModel`) is not yet implemented in runtime config workflow.
- Dataset validation exists (`src/od_training/dataset/convert.py`) but is not yet a mandatory training gate.

## Notes
- Historical planning artifact `docs/dev/plan.md` was retired.
- Historical prompt artifact `docs/dev/research_prompts.md` was removed.
- Use `docs/dev/progress.md` for dated change history.
