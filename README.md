# od_training

Personal object-detection training toolkit for:

- dataset preparation/export around FiftyOne
- YOLO and RF-DETR training entrypoints
- inference helpers
- Roboflow utility operations

The repository is packaged and exposes a unified CLI command: `odt`.

## Quick Start

1. install in editable mode:

```bash
python -m pip install -e .
```

2. verify environment:

```bash
odt utility verify-env
```

3. inspect CLI surface:

```bash
odt --help
```

## Main Entrypoints

- preferred: `odt <group> <command> [args...]`
- compatibility wrapper: `python scripts/odt.py <group> <command> [args...]`

Top-level dispatch lives in `src/od_training/cli/main.py`.

## Repository Layout

- `src/od_training/`
  - package source modules
- `scripts/`
  - wrapper and utility scripts
- `test/`
  - unit and integration tests
- `docs/CLI.md`
  - CLI contract
- `docs/context/`
  - workflow and handoff context docs

## Configuration

Roboflow-sensitive settings are loaded from a local config JSON.

Resolution order:

1. `ODT_CONFIG_PATH`
2. repo-local `config/local_config.json` (if present)
3. `~/.config/od_training/local_config.json` (or `$XDG_CONFIG_HOME/od_training/local_config.json`)

`config/local_config.json` is gitignored.

## Common Commands

Dataset manage/export:

```bash
odt dataset manage --dataset-dir data/dummy_yolo/data.yaml --name demo_ds --split --export-dir runs/export_demo
```

Config helpers:

```bash
odt utility config-init
odt utility config-show --mask-secrets
```

YOLO training:

```bash
odt train yolo --model yolo11n.pt --data runs/export_demo/dataset.yaml --epochs 10 --batch 8
```

By default, training commands run dataset preflight validation first.
Use `--no-validate-data` to bypass preflight when intentionally needed.

RF-DETR training:

```bash
odt train rfdetr --dataset runs/export_demo --model rfdetr_nano --epochs 10 --batch 4
```

Inference:

```bash
odt infer run --type yolo --model yolo11n.pt --source data/images --save-dir runs/infer_demo
```

## dst -> od_training Handoff

This repo consumes curated datasets produced by `dst`.

Contract document:
- `docs/context/dst_handoff_contract.md`

Use this when exporting from `dst` and importing/training in `od_training`.
