# Development Status

**Last updated:** 2026-02-14
**Purpose:** Current, repo-local status of implemented functionality and open work.

## Implemented

### CLI And Structure
- Unified CLI entrypoint: packaged `odt` command (`pyproject.toml`)
- Compatibility wrapper: `scripts/odt.py`
- Modular dispatch: `src/od_training/cli/main.py`
- Command groups live under:
  - `src/od_training/dataset/`
  - `src/od_training/train/`
  - `src/od_training/infer/`
  - `src/od_training/utility/`

### Dataset Pipeline
- YOLO import/split/augment/export pipeline implemented in `src/od_training/dataset/manager.py`
- Export defaults to zero-copy symlink mode (`export_media="symlink"`) with `--copy-images` for portable copies
- Label field for export is configurable (`--label-field`)
- Class discovery field for export is configurable (`--classes-field`)
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
- Runtime config with portable path resolution and env fallback: `src/od_training/utility/runtime_config.py`
- Explicit config CLI commands:
  - `odt utility config-init`
  - `odt utility config-show`
- Environment verification: `src/od_training/utility/verify_env.py`
- Roboflow download/upload tooling:
  - `src/od_training/utility/roboflow_download.py`
  - `src/od_training/utility/roboflow_upload.py`

### Cross-Repo Compatibility
- Shared dependency baseline with `dst` documented in:
  - `docs/context/cross_repo_compatibility_matrix.md`
- Handoff contract documented in:
  - `docs/context/dst_handoff_contract.md`

## In Progress / Pending
- No blocking pending architecture items in the current harmonization scope.

## Notes
- Historical planning artifact `docs/dev/plan.md` was retired.
- Historical prompt artifact `docs/dev/research_prompts.md` was removed.
- Use `docs/dev/progress.md` for dated change history.
