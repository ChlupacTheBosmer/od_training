# Agent Context Space

Purpose: onboarding context for AI agents working in this repository.

This folder is the agent-first launchpad. It does not duplicate full project docs; it routes to canonical sources and defines execution standards so changes stay consistent with repository structure.

## Mandatory Read Order

1. `docs/agent/context_index.yaml`
2. `docs/agent/engineering_contract.md`
3. `docs/agent/docs_guide.md`
4. `docs/agent/first_task_protocol.md`
5. Canonical project docs:
   - `README.md`
   - `docs/dev/status.md`
   - `docs/dev/progress.md`
   - `docs/CLI.md`

## Fast Bootstrap (Recommended)

1. Check current worktree and repository state.
2. Confirm environment and dependency imports:
   - `odt utility verify-env`
3. Route to the relevant code area using `context_index.yaml` and `README.md` maps.
4. Implement changes in `src/od_training/*` by default.
5. Run targeted tests.
6. Update docs impacted by the change (see `engineering_contract.md`).

## Scope Notes

- Use `docs/dev/status.md` as current implementation truth.
- Use `docs/dev/progress.md` for change chronology.
- Treat `docs/dev/audit_report.md`, `docs/dev/discovery.md`, and `docs/dev/research.md` as historical/supporting context.
- Use `docs/src/*` for module-level reference snapshots.
- `external/` holds local clones for upstream code inspection only; policy and repo list live in `external/README.md`.

## Compatibility Layer Warning

The repository includes legacy compatibility modules in `src/` (`runtime_config.py`, `cli_utils.py`, `device_utils.py`, `converters.py`) that mirror active logic under `src/od_training/*`. If a change touches shared behavior, confirm both sides remain aligned.
