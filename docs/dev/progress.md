# Development Changelog

This document tracks major repository-level development milestones in chronological order.

## 2026-02-11

### Repository Audit Baseline
- Audited portability and deployment readiness for Kubernetes-oriented workflows.
- Added environment-variable fallback for Roboflow credentials in runtime config handling.
- Updated pipeline verification script behavior to run with or without a local `venv`.
- Applied dependency compatibility guardrail: `scipy<1.13.0`.
- Published historical snapshot: [`./audit_report.md`](./audit_report.md).

### Post-Audit Structural Consolidation
- Introduced unified CLI entrypoint (`scripts/odt.py`) and modular dispatcher (`src/od_training/cli/main.py`).
- Consolidated features under the `src/od_training/` package (`dataset`, `train`, `infer`, `utility`).
- Added/validated dataset export behavior:
  - default symlink export mode for zero-copy image handling
  - optional copy-based portable export
  - COCO sidecar exports per split
- Added shared utility modules for device resolution and CLI unknown-arg parsing.
- Added/expanded test coverage for dataset export and one-process pipeline flows.

### Documentation Refactor
- Retired obsolete planning/prompt artifacts from active dev docs.
- Added current status source: [`./status.md`](./status.md).
- Curated research notes and replaced legacy training notes with cookbook-style guidance.

## 2026-02-14

### Packaging and Portability
- Added `pyproject.toml` with console-script entrypoint `odt`.
- Converted `scripts/odt.py` into a compatibility wrapper without `sys.path` injection.
- Switched tests/imports to package namespace (`od_training.*`) to match packaged execution.

### Config Hardening
- Refactored runtime config resolution to:
  1. `ODT_CONFIG_PATH`
  2. repo-local `config/local_config.json` (if present)
  3. `~/.config/od_training/local_config.json`
- Added optional `--config` argument to Roboflow upload/download utilities.
- Removed import-time `ensure_local_config()` side effects from dataset/train/infer/utility modules.

### Dataset Export Contract
- Added `--label-field` and `--classes-field` to `odt dataset manage`.
- Export pipeline now supports non-`ground_truth` fields for YOLO/COCO export.

### Cross-Repo Harmonization
- Aligned FiftyOne compatibility to `fiftyone>=1.11,<2.0`.
- Added shared stack documents:
  - `docs/context/cross_repo_compatibility_matrix.md`
  - `docs/context/dst_handoff_contract.md`

### Next-Phase Hardening
- Added typed runtime config models in `src/od_training/utility/runtime_config.py`:
  - `LocalRuntimeConfig`
  - `RoboflowRuntimeConfig`
- Added explicit config utility commands:
  - `odt utility config-init`
  - `odt utility config-show --mask-secrets`
- Added preflight training validators:
  - `src/od_training/train/preflight.py`
  - wired into `odt train yolo` and `odt train rfdetr`
- New train CLI safety flags:
  - `--no-validate-data`
  - `--fail-on-validation-warnings`
- Added unit coverage for typed config parsing and preflight validation behavior.
