# od_training

Personal utility repository for object detection training workflows.

This is a working toolbox for:
- preparing datasets
- training YOLO and RF-DETR models
- running inference
- handling a few Roboflow utility tasks

## Quick Start (Re-entry)

If you return after a long time, do this first:
1. Read current project state: `docs/dev/status.md`
2. Read recent changes: `docs/dev/progress.md`
3. Check CLI surface: `docs/CLI.md`
4. Confirm environment: `venv/bin/python scripts/odt.py utility verify-env`
5. Check local config: `config/local_config.json`

## Main Entrypoint

Use the unified CLI:

```bash
python scripts/odt.py <group> <command> [args...]
```

Groups:
- `dataset`
- `train`
- `infer`
- `utility`

Top-level dispatch lives in `src/od_training/cli/main.py`.

## Repository Map

- `scripts/odt.py`: unified CLI wrapper
- `scripts/verify_pipeline.sh`: end-to-end smoke script
- `src/od_training/dataset/`: import/split/augment/export/convert/view logic
- `src/od_training/train/`: YOLO and RF-DETR training wrappers
- `src/od_training/infer/`: inference runner
- `src/od_training/utility/`: runtime config, env checks, Roboflow helpers
- `docs/CLI.md`: command reference
- `docs/agent/`: AI-agent onboarding context pack
- `docs/dev/`: developer notes (`status`, `progress`, research/history)
- `docs/src/`: per-module documentation snapshots
- `test/`: unit/integration tests

## Where To Look For What

| Need | Start Here |
|---|---|
| Dataset import/augmentation/export | `src/od_training/dataset/manager.py` |
| Format conversion (YOLO/COCO) | `src/od_training/dataset/convert.py` |
| YOLO training | `src/od_training/train/yolo.py` |
| RF-DETR training | `src/od_training/train/rfdetr.py` |
| Inference | `src/od_training/infer/runner.py` |
| Roboflow download | `src/od_training/utility/roboflow_download.py` |
| Roboflow upload | `src/od_training/utility/roboflow_upload.py` |
| Runtime config and credential resolution | `src/od_training/utility/runtime_config.py` |
| Environment diagnostics | `src/od_training/utility/verify_env.py` |
| CLI command routing | `src/od_training/cli/main.py` |

## Common Command Patterns

Dataset workflow:

```bash
python scripts/odt.py dataset manage --dataset-dir <data.yaml-or-dir> --name <dataset_name> --split --export-dir <export_dir>
```

YOLO training:

```bash
python scripts/odt.py train yolo --model yolo11n.pt --data <export_dir>/dataset.yaml --epochs 100 --batch 16 --imgsz 640
```

RF-DETR training:

```bash
python scripts/odt.py train rfdetr --dataset <dataset_root> --model rfdetr_nano --epochs 50 --batch 8 --lr 1e-4
```

Inference:

```bash
python scripts/odt.py infer run --type yolo --model <weights.pt> --source <image|dir|video|0|rtsp://...> --save-dir <out_dir>
```

## Config And Environment

Local config file:
- `config/local_config.json`

Roboflow credentials can be resolved from:
1. CLI arguments
2. `config/local_config.json`
3. Environment variables:
   - `ROBOFLOW_API_KEY`
   - `ROBOFLOW_WORKSPACE`
   - `ROBOFLOW_PROJECT`

Dependencies are pinned/ranged in `requirements.txt`.

## Notes On Layout

- `src/runtime_config.py`, `src/cli_utils.py`, `src/device_utils.py`, and `src/converters.py` exist as compatibility-layer modules mirroring the modular package under `src/od_training/`.
- New work should generally target modules under `src/od_training/`.
