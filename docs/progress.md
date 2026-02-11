# Project Progress

## Phase 1: Initialization & Discovery
- [x] Create project structure
- [x] Initialize virtual environment
- [x] Create documentation directory
- [/] Complete discovery phase

## Phase 2: Research & Planning
- [x] Research YOLOv11 vs YOLO26
- [x] Research data interchange (YOLO <-> COCO)
- [x] Research RF-DETR workflow
- [x] Research Dependency Compatibility (RF-DETR + Ultralytics)

## Phase 3: Audit Fix Implementation (Feb 2026)

All audit fixes implemented in a single pass across 5 severity phases.

### Critical Fixes
- [x] `dataset_manager.py`: Replaced broken `export_media=False` with `export_media="symlink"`. FiftyOne now creates symlinks to original images and auto-generates `dataset.yaml`. Removed `setup_absolute_yaml()` and `.txt` listing files. Fixed `copy_images=True` path (was `pass`). Added `--include-confidence` flag. Fixed COCO JSON `file_name` dedup with `_fix_coco_filenames()`.
- [x] `inference.py`: Moved `RFDETR_MAP` inside try/except block to prevent `NameError` when rfdetr isn't installed.
- [x] `train_rfdetr.py`: Improved checkpoint error handler — now verifies at least one `.pth` exists in output dir before silencing the error.

### High Severity Fixes
- [x] Created `src/device_utils.py` — unified `resolve_device()` (explicit → env var → torch probe → cpu).
- [x] Created `src/cli_utils.py` — shared `parse_unknown_args()` with proper type conversion (bool, int, float, comma-separated tuples).
- [x] `train_rfdetr.py`: Integrated shared modules, added `--device` CLI arg.
- [x] `train_yolo.py`: Replaced ad-hoc kwargs parsing loop with shared module.
- [x] `src/converters.py`: Full rewrite — globox fallback for COCO→YOLO, `ValidationReport` namedtuple, validates 5+6 column labels, bbox ranges, class ID bounds.
- [x] `verify_env.py`: Tracks hard failures, exits with code 1, categorized rfdetr/roboflow as optional (WARN).
- [x] `requirements.txt`: Pinned wide compatible ranges.

### Medium Severity Fixes
- [x] `inference.py` UX: Video FPS from source instead of hardcoded 30, original filenames in directory mode, `--save-dir` default=None.
- [x] `upload_weights.py`: Added `ROBOFLOW_API_KEY` env var fallback, preflight checks (file exists + non-empty), lazy roboflow import.
- [x] `verify_pipeline.sh`: Complete rewrite — tests symlink export, checks 5-column labels, `--no-clean` flag.

### Low Severity / Structural
- [x] Removed unused imports (`shutil`, `yaml`, `foz`, `np`, `Path`) from `dataset_manager.py` and `train_rfdetr.py`.
- [x] Hardened `sys.path` with `Path(__file__).resolve().parent.parent` in both training scripts.
