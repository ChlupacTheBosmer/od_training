# Test Implementation Plan

## 1) Scope and Objective
Build a comprehensive automated test suite for this repository, with:
- Unit tests for all core business logic and branch behavior
- Integration tests for dataset export, training, and inference
- Model-specific end-to-end smoke tests for both YOLO and RF-DETR
- One-process orchestration tests that run export -> train -> inference in a single Python test process

This plan is based on the current code in `src/` and `scripts/`.

## 2) Current Architecture (What Connects to What)
- `src/od_training/train/yolo.py`
  - Uses `src.device_utils.resolve_device`
  - Uses `src.cli_utils.parse_unknown_args`
  - Wraps `ultralytics.YOLO(...).train(...)` and optional `model.export(...)`
- `src/od_training/train/rfdetr.py`
  - Uses `src.device_utils.resolve_device`
  - Uses `src.cli_utils.parse_unknown_args`
  - Resolves architecture via `MODEL_MAP`, then calls `model.train(...)`
- `src/od_training/infer/runner.py`
  - YOLO path: `YOLO(model).predict(...)` + supervision annotators
  - RF-DETR path: `load_rfdetr_model(...)` + `model.predict(...)`
  - Handles source as image file, image directory, video file, webcam/RTSP
- `src/od_training/dataset/manager.py`
  - Imports/loads YOLO dataset via FiftyOne
  - Optional split and augmentation
  - Export path generates:
    - YOLO export (`images/<split>`, `labels/<split>`, `dataset.yaml`)
    - COCO export (`_annotations_<split>.coco.json`)
- `src/converters.py`
  - YOLO <-> COCO conversion wrappers
  - `validate_dataset(...)` for image+label integrity
- `src/od_training/dataset/convert_cli.py`
  - CLI wrapper over `src.converters`
- `src/od_training/utility/roboflow_upload.py`
  - Preflight checks + Roboflow SDK deploy

## 3) Proposed Test Layout
Use `pytest` and keep all tests under `test/`:

- `test/unit/`
  - `test_cli_utils.py`
  - `test_device_utils.py`
  - `test_converters.py`
  - `test_dataset_manager.py`
  - `test_train_yolo.py`
  - `test_train_rfdetr.py`
  - `test_inference.py`
  - `test_upload_weights.py`
  - `test_convert_format_cli.py`
- `test/integration/`
  - `test_dataset_export_smoke.py`
  - `test_pipeline_yolo_one_process.py`
  - `test_pipeline_rfdetr_one_process.py`
  - `test_pipeline_parametrized_one_process.py`
- `test/fixtures/`
  - lightweight dummy image/label generators
  - helper builders for YOLO and COCO folder structures

## 4) Test Infrastructure Decisions
- Framework: `pytest`
- Mocking: `pytest-mock` (or `unittest.mock`)
- Markers:
  - `@pytest.mark.unit`
  - `@pytest.mark.integration`
  - `@pytest.mark.slow`
  - `@pytest.mark.gpu` (optional)
- Default local command:
  - `pytest -m "unit"`
- Full smoke command:
  - `pytest -m "integration and slow"`

## 5) Unit Test Plan (Detailed)

### 5.1 `src/cli_utils.py`
Target: `parse_unknown_args`, `_convert_value`

Cases:
- booleans: `true/false`
- `none` -> `None`
- int/float conversion
- comma-separated tuple conversion (including mixed typed values)
- bare flags become `True`
- dash-to-underscore key conversion
- ignores non-flag stray tokens safely

### 5.2 `src/device_utils.py`
Target: `resolve_device`

Cases:
- explicit device always wins
- `CUDA_VISIBLE_DEVICES` set -> `"cuda"`
- torch available + cuda available -> `"cuda"`
- torch import failure -> `"cpu"`
- no env, no cuda -> `"cpu"`

### 5.3 `src/converters.py`
Target: conversion wrappers + validation

Cases:
- `convert_yolo_to_coco`: calls globox API correctly
- `convert_coco_to_yolo`: ultralytics primary path
- `convert_coco_to_yolo`: ultralytics unavailable -> globox fallback
- `validate_dataset`:
  - corrupt image -> error
  - missing label -> warning
  - invalid column count -> error
  - 6-column labels -> warning (not error)
  - non-numeric values -> error
  - out-of-range bbox values -> error
  - class index out of bounds when `class_names` provided -> error

### 5.4 `src/od_training/dataset/manager.py`
Targets: split, augmentation, export, COCO filename normalization

Cases:
- `load_or_create_dataset`:
  - load existing dataset path
  - import from directory
  - import from yaml path passed as file
  - split happens only when split tags missing
- `augment_samples`:
  - no matching samples -> early return
  - destination dataset creation/loading logic
  - augmented sample gets copied tags + `"augmented"`
- `export_pipeline`:
  - `copy_images=False` sets symlink mode
  - `copy_images=True` sets copy mode
  - per-split export calls happen with expected args
  - optional `export_tags` filtering uses ALL tags logic
- `_fix_coco_filenames`:
  - normalizes to basenames when unique
  - preserves full paths when duplicates detected

### 5.5 `src/od_training/train/yolo.py`
Target: `train_yolo`

Cases:
- ClearML init success and failure fallback
- model instantiated with requested weights
- if model name contains `yolo26` and no optimizer provided -> sets `MuSGD`
- explicit optimizer in kwargs is preserved
- `model.train` called with merged args
- TensorRT export is triggered only when `device != cpu` and `epochs > 1`
- TensorRT export errors are handled non-fatally

### 5.6 `src/od_training/train/rfdetr.py`
Target: `train_rfdetr`

Cases:
- known model key resolves from `MODEL_MAP`
- unknown model key fallback contains map key substring
- unknown/unmatched key defaults to `RFDETRMedium`
- missing dataset path raises `FileNotFoundError`
- train args include resolved device and override kwargs
- checkpoint-best missing `FileNotFoundError` branch:
  - if `.pth` exists -> tolerated
  - if no `.pth` exists -> re-raises

### 5.7 `src/od_training/infer/runner.py`
Targets: `load_rfdetr_model`, `run_inference`

Cases for `load_rfdetr_model`:
- RF-DETR unavailable -> ImportError
- known map key returns matching class
- nonexistent path + inferred key fallback
- nonexistent path no fallback -> FileNotFoundError
- existing custom weight path with explicit `rfdetr_arch`
- existing custom weight path with inferred arch from filename

Cases for `run_inference`:
- unsupported `model_type` -> ValueError
- source is image file -> one annotated output file
- source is directory -> output names use `<original>_annotated.jpg`
- source invalid/unreadable -> ValueError
- video path initializes writer and writes output video
- `show=False` path avoids GUI dependency

### 5.8 `src/od_training/utility/roboflow_upload.py`
Target: `upload_weights`

Cases:
- weights path missing -> FileNotFoundError
- empty weights file -> ValueError
- roboflow import missing -> ImportError
- successful deploy calls expected SDK methods with expected params
- deploy failure propagates exception

### 5.9 `src/od_training/dataset/convert_cli.py`
Target: CLI dispatch

Cases:
- `yolo2coco` without `--images` exits with code 1
- `yolo2coco` calls `convert_yolo_to_coco`
- `coco2yolo` calls `convert_coco_to_yolo`

## 6) Integration Test Plan

### 6.1 Dataset Export Smoke
File: `test/integration/test_dataset_export_smoke.py`

Flow:
1. Build tiny YOLO dataset fixture in temp dir
2. Call `load_or_create_dataset(..., split_ratios=...)`
3. Call `augment_samples(..., filter_tags=["train"], num_aug=1)`
4. Call `export_pipeline(..., copy_images=False)`
5. Assert:
   - `dataset.yaml` exists
   - `images/train` exists and contains symlinks
   - `labels/train/*.txt` exists and all labels have 5 columns
   - `_annotations_train.coco.json` exists and includes images/annotations

### 6.2 YOLO One-Process Pipeline Smoke
File: `test/integration/test_pipeline_yolo_one_process.py`

Flow in one test function/process:
1. Create/export dataset (same as above)
2. Run `train_yolo(..., epochs=1, batch=1, imgsz=64, device="cpu")`
3. Locate produced YOLO weights (`best.pt` or fallback `last.pt`)
4. Run `run_inference(..., model_type="yolo", save_dir=...)`
5. Assert annotated output image(s) exist and are non-empty

### 6.3 RF-DETR One-Process Pipeline Smoke
File: `test/integration/test_pipeline_rfdetr_one_process.py`

Flow in one test function/process:
1. Create/export dataset with COCO files
2. Run `train_rfdetr(..., epochs=1, batch_size=1, device="cpu", output_dir=<tmp>)`
3. Locate produced `.pth` checkpoint
4. Run `run_inference(..., model_type="rfdetr", model_name=<checkpoint>, rfdetr_arch="rfdetr_nano")`
5. Assert annotated output image(s) exist and are non-empty

### 6.4 Parametrized Unified Pipeline
File: `test/integration/test_pipeline_parametrized_one_process.py`

Single parametrized test over `["yolo", "rfdetr"]`:
- Executes shared dataset export first
- Branches to model-specific training/inference
- Ensures each model can complete export -> train -> inference sequentially in one process

## 7) Execution Strategy and Runtime Control
- Fast path (default): run only unit tests with mocked model frameworks
- Slow path: gated by environment variable, e.g. `RUN_MODEL_SMOKE=1`
- Use markers so CI and local runs can choose:
  - `pytest -m unit`
  - `RUN_MODEL_SMOKE=1 pytest -m "integration and slow"`

## 8) Known Risk Areas to Explicitly Test
- RF-DETR arch inference from custom checkpoint filename
- YOLO26 optimizer autoselection behavior
- COCO filename normalization when duplicate basenames exist
- `include_confidence=True` behavior for export labels
- handling absence of optional dependencies (RF-DETR/Roboflow)

## 9) Suggested Implementation Phases
Phase 1:
- Add pytest scaffolding under `test/`
- Implement all pure unit tests (`src/*`, utility scripts)

Phase 2:
- Add dataset export integration smoke
- Replace shell-only checks from `scripts/verify_pipeline.sh` with pytest equivalent

Phase 3:
- Add model one-process smoke tests for YOLO and RF-DETR
- Gate heavy tests via marker/env variable

Phase 4:
- Stabilize flakiness and tune tiny dataset fixtures for repeatability

## 10) Definition of Done
- Unit coverage includes all key branches in listed modules
- Integration suite includes at least:
  - 1 dataset export smoke test
  - 1 YOLO one-process pipeline test
  - 1 RF-DETR one-process pipeline test
- All tests runnable from repository root with deterministic pass/fail signals
