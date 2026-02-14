# dst -> od_training Handoff Contract

This document defines the expected interface between dataset curation (`dst`)
and model training (`od_training`).

## Required Inputs

- FiftyOne dataset name
- detection field to export (default: `ground_truth`)
- split tags used by curated samples (`train`, `val`, `test` unless overridden)
- export media mode (`symlink` or `copy`)

## Canonical Flow

1. Curate data in `dst` and finalize the label field to train from.
2. Export from `dst` with the same split tags expected by training.
3. Load/export in `od_training` using matching field/tag arguments.
4. Train with the exported artifact (`dataset.yaml` for YOLO; COCO sidecars for RF-DETR).

## Export Contract

`od_training dataset manage` expects:

- YOLO layout:
  - `images/train|val|test/`
  - `labels/train|val|test/`
  - `dataset.yaml`
- COCO sidecars (for RF-DETR workflows):
  - `_annotations_train.coco.json`
  - `_annotations_val.coco.json`
  - `_annotations_test.coco.json`

## CLI Mapping

- `dst` export source field should map to:
  - `odt dataset manage --label-field <field>`
- class discovery field should map to:
  - `odt dataset manage --classes-field <field>.detections.label`
- split tags should map to:
  - `--train-tag`, `--val-tag`, `--test-tag`
- media mode should map to:
  - default symlink mode, or `--copy-images` for portable exports
- confidence column handling should map to:
  - default 5-column YOLO export
  - `--include-confidence` when downstream tooling expects 6-column YOLO labels

## Copy-Paste Procedure

```bash
# 1) Export curated data from FiftyOne via od_training dataset manage
odt dataset manage \
  --name <fiftyone_dataset_name> \
  --export-dir runs/export/<run_name> \
  --label-field <curated_label_field> \
  --train-tag <train_tag> --val-tag <val_tag> --test-tag <test_tag>

# Optional: portable copy mode
#   add --copy-images
#
# Optional: include confidence column in YOLO labels
#   add --include-confidence

# 2) Train YOLO using exported YAML
odt train yolo \
  --data runs/export/<run_name>/dataset.yaml \
  --model yolo11n.pt \
  --epochs 50 --batch 16

# 3) Train RF-DETR using exported split layout + COCO sidecars
odt train rfdetr \
  --dataset runs/export/<run_name> \
  --model rfdetr_nano \
  --epochs 50 --batch 8
```

## Guardrails

- Use one dataset + label field per training run for reproducibility.
- Keep split tags explicit when non-default names are used.
- Do not mix corrected and uncorrected fields implicitly; pass `--label-field`.
