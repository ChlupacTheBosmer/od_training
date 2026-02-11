# Research Notes (Curated)

**Last curated:** 2026-02-11  
**Scope:** keep only repo-relevant, source-backed findings that inform implementation decisions.

## 1. What Is Confirmed In This Repo

- Unified CLI is active via `scripts/odt.py` and dispatches into `src/od_training/cli/main.py`.
- Dataset management/export lives in `src/od_training/dataset/manager.py`.
- Current zero-copy default is symlink export (`export_media="symlink"`), with `--copy-images` for full portable copies.
- Runtime config (with env fallbacks) is implemented in `src/od_training/utility/runtime_config.py`.

## 2. Ultralytics (YOLO) Findings

### 2.1 Model Lineup And Positioning
- Ultralytics docs currently present YOLO11 as a production model and also document YOLO26 as a newer generation.
- YOLO26 docs describe end-to-end design goals and list MuSGD among key training innovations.

### 2.2 Training Knobs Worth Using
From Ultralytics training/config docs:
- `freeze` supports freezing layers (integer or layer index list).
- `close_mosaic` is available to disable mosaic augmentation during final epochs.
- `optimizer=auto` is available and often preferred as a baseline.
- Full train config includes standard controls like `imgsz`, `batch`, `lr0`, `weight_decay`, `patience`.

Repo implication:
- Keep CLI pass-through for advanced args (already supported in train wrappers).
- Avoid hardcoding overly opinionated defaults beyond safe baseline behavior.

### 2.3 Label Path Resolution Constraint
Ultralytics dataset handling still derives label paths from image paths (`images/` -> `labels/`) via internal label-path mapping.

Repo implication:
- Export layout must preserve YOLO image/label directory symmetry.
- The current FiftyOne symlink export strategy in `src/od_training/dataset/manager.py` is aligned with this constraint.

## 3. RF-DETR Findings

### 3.1 Data Contract
Official RF-DETR docs state support for both COCO and YOLO formats with auto-detection, and define expected split structures.

Repo implication:
- Our COCO sidecar export per split is still useful and compatible.
- For RF-DETR training entrypoints, keep explicit dataset structure expectations documented.

### 3.2 Training Controls
RF-DETR docs include the following practical controls:
- `batch_size`, `grad_accum_steps`, and recommendation to target effective batch around 16.
- `early_stopping`, `checkpoint_interval`, and `resume` options.
- Logging controls for TensorBoard and Weights & Biases.

Repo implication:
- Keep unknown-arg forwarding enabled in RF-DETR CLI wrapper for advanced control.
- Preserve explicit checkpoint handling safeguards already present in training wrapper.

### 3.3 Checkpoints
RF-DETR docs identify multiple checkpoint outputs including EMA/regular/best variants.

Repo implication:
- Keep "best checkpoint missing" handling tolerant when other valid checkpoints exist.

## 4. Albumentations Findings Relevant To This Repo

- Bounding box-aware pipelines require `A.BboxParams(...)` with a format matching your labels.
- Albumentations docs recommend bbox-safe transforms and controls such as:
  - `min_area`
  - `min_visibility`
  - bbox-safe crop transforms (`BBoxSafeRandomCrop`, `AtLeastOneBBoxRandomCrop`)

Repo implication:
- In `src/od_training/dataset/manager.py`, current augmentation can be strengthened by replacing unconstrained random crop with bbox-safe variants for detection tasks.

## 5. Dependency Compatibility (Curated)

Use repo requirements as source of truth:
- `numpy>=1.23.5,<2.0.0`
- `scipy<1.13.0`
- `pydantic>=1.10,<3.0.0`
- `ultralytics>=8.0`
- `rfdetr[metrics]`

Important cleanup:
- Any older recommendation of `pydantic<2.0.0` is stale relative to current project constraints and should not be used as canonical guidance here.

## 6. Decisions To Keep

- Keep symlink as default export mode for zero-copy workflows.
- Keep `--copy-images` for portable artifact creation.
- Keep confidence-stripping as default export behavior (5-column labels), with opt-in `--include-confidence` when needed for curation workflows.

## 7. Sources

Primary sources used for this curated pass:
- Ultralytics docs home: https://docs.ultralytics.com/
- Ultralytics YOLO26 page: https://docs.ultralytics.com/models/yolo26/
- Ultralytics train mode docs: https://docs.ultralytics.com/modes/train/
- Ultralytics configuration docs: https://docs.ultralytics.com/usage/cfg/
- Ultralytics dataset utility reference (label path behavior): https://docs.ultralytics.com/reference/data/utils/
- RF-DETR docs home: https://rfdetr.roboflow.com/
- RF-DETR dataset formats: https://rfdetr.roboflow.com/data-formats/index.html
- RF-DETR API/training parameters: https://rfdetr.roboflow.com/model/
- RF-DETR advanced training docs: https://rfdetr.roboflow.com/advanced-usage/training/
- Albumentations bbox guide: https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/
- Albumentations API (bbox-safe transforms): https://albumentations.ai/docs/api-reference/albumentations/augmentations/crops/transforms/
