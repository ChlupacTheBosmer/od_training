# Training Cookbook

**Last curated:** 2026-02-11  
**Audience:** engineers running repeatable YOLO and RF-DETR training in this repository.

## 1. Purpose And Operating Model

This cookbook is a methodology guide, not a one-off experiment note.

Use it to run a repeatable loop:
1. Prepare and validate dataset.
2. Train a conservative baseline.
3. Tune one variable block at a time.
4. Evaluate on fixed validation/test sets.
5. Promote only reproducible improvements.

Primary docs used for this guide are listed in [Sources](#9-sources).

## 2. Repository Commands (Canonical)

### 2.1 Dataset Preparation
```bash
odt dataset manage \
  --dataset-dir <path/to/data.yaml-or-dataset-dir> \
  --name <dataset_name> \
  --split \
  --export-dir <export_dir>
```

Defaults:
- Export mode is symlink (`export_media="symlink"`) for zero-copy media handling.
- Use `--copy-images` when you need portable, self-contained artifacts.

### 2.2 YOLO Training
```bash
odt train yolo \
  --model yolo11n.pt \
  --data <export_dir>/dataset.yaml \
  --epochs 100 --batch 16 --imgsz 640
```

### 2.3 RF-DETR Training
```bash
odt train rfdetr \
  --dataset <rfdetr_dataset_root> \
  --model rfdetr_nano \
  --epochs 50 --batch 8 --lr 1e-4
```

### 2.4 Inference Smoke
```bash
odt infer run \
  --type yolo \
  --model <weights.pt> \
  --source <image-or-dir-or-video> \
  --save-dir <output_dir>
```

## 3. Dataset Contract And Quality Gates

### 3.1 Keep Splits Stable
- Do not reshuffle train/val/test between tuning runs.
- Make one baseline split and keep it fixed for comparability.

### 3.2 Validate Before Training
The repo includes `validate_dataset()` in `src/od_training/dataset/convert.py`.

Recommended checks before each major run:
- Corrupt images
- Missing labels
- Invalid bbox ranges
- Unexpected column counts

### 3.3 Export Strategy
- Default (`symlink`) is preferred for iterative internal work.
- `--copy-images` is preferred for handoff, archival, or cross-volume portability.

## 4. YOLO Methodology

Ultralytics train/config docs expose the knobs below; this section turns them into a stable workflow.

### 4.1 Baseline Recipe
Start here before advanced tuning:
- `imgsz=640`
- `batch` as large as stable on your GPU
- `optimizer=auto`
- default augmentation settings
- no custom freeze initially

### 4.2 High-Value Tuning Order
Tune in this order to reduce confounding:
1. Capacity and resolution:
- model size (`n/s/m/l/x`)
- `imgsz`
2. Optimization:
- `lr0`, weight decay, scheduler-related settings
3. Regularization/augmentation:
- `close_mosaic`
- `mixup`
- class/instance balance actions in the dataset
4. Transfer behavior:
- `freeze` (int/list) for small-data finetuning

### 4.3 Practical Guidance
- Use `close_mosaic` when late-epoch instability appears.
- Use `freeze` for small datasets where full-backbone updates overfit.
- Keep one experiment variable block per run (avoid changing model size + LR + augmentation together).

## 5. RF-DETR Methodology

RF-DETR docs provide explicit training controls. Use them systematically.

### 5.1 Baseline Recipe
- Dataset root with expected split structure
- `batch_size` + `grad_accum_steps` targeting effective batch around 16
- conservative epochs and LR baseline first

### 5.2 High-Value Tuning Order
1. Effective batch stability:
- adjust `batch_size`
- adjust `grad_accum_steps`
2. Duration and stopping:
- `epochs`
- `early_stopping`
3. Recovery and reproducibility:
- `checkpoint_interval`
- `resume`

### 5.3 Logging And Checkpointing
From RF-DETR docs:
- TensorBoard and W&B logging are first-class options.
- Multiple checkpoints (EMA/regular/best) are produced.

Repo alignment:
- `src/od_training/train/rfdetr.py` already forwards unknown args, so advanced controls can be passed through CLI.

## 6. Augmentation Playbook (Detection-Safe)

Albumentations is powerful, but object detection requires bbox-safe configuration.

### 6.1 Required Setup
- Always define `A.BboxParams(...)` with matching bbox format.
- Enforce filtering with `min_area` and `min_visibility`.

### 6.2 Crop Policy
For detection tasks, prefer bbox-safe crops over unconstrained random crop:
- `A.BBoxSafeRandomCrop`
- `A.AtLeastOneBBoxRandomCrop`

### 6.3 Recommended Rollout Pattern
1. Start with geometric + photometric baseline (flip, brightness/contrast).
2. Add one crop policy change.
3. Compare mAP and failure modes before adding more transforms.

## 7. Experiment Tracking Rules

Even when not all native integrations are enabled, keep discipline:
- Fixed run naming convention: `<model>_<dataset>_<date>_<variant>`
- Persist CLI args per run
- Store evaluation outputs under deterministic directories
- Keep a single "best baseline" checkpoint for each dataset-model pair

Repo note:
- Training wrappers initialize ClearML when available.
- RF-DETR docs additionally support TensorBoard/W&B arguments.

## 8. Common Failure Modes

### 8.1 Metrics Jump But Visual Quality Regresses
- Confirm validation split did not drift.
- Check augmentation is not too aggressive late in training.

### 8.2 RF-DETR Instability
- Re-check effective batch size (`batch_size * grad_accum_steps * num_gpus`).
- Use early stopping and checkpoint intervals for safer long runs.

### 8.3 YOLO Label Mismatch Issues
- Ensure dataset layout preserves `images/<split>` and `labels/<split>` pairing.
- Use repo export defaults unless you intentionally change structure.

## 9. Sources

Primary sources used in this cookbook:
- Ultralytics docs home: https://docs.ultralytics.com/
- Ultralytics train mode: https://docs.ultralytics.com/modes/train/
- Ultralytics configuration reference: https://docs.ultralytics.com/usage/cfg/
- Ultralytics YOLO26 model page: https://docs.ultralytics.com/models/yolo26/
- Ultralytics data utils reference: https://docs.ultralytics.com/reference/data/utils/
- RF-DETR docs home: https://rfdetr.roboflow.com/
- RF-DETR model/API docs: https://rfdetr.roboflow.com/model/
- RF-DETR data formats: https://rfdetr.roboflow.com/data-formats/index.html
- RF-DETR advanced training: https://rfdetr.roboflow.com/advanced-usage/training/
- Albumentations bbox guide: https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/
- Albumentations crop transforms: https://albumentations.ai/docs/api-reference/albumentations/augmentations/crops/transforms/
