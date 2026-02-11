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
