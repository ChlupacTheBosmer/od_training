# Engineering Contract For AI Agents

This contract defines repository-specific standards that agents must follow.

## 1. Architectural Contract

- Primary code namespace is `src/od_training/`.
- Unified CLI entrypoint is `scripts/odt.py`.
- CLI dispatch map is in `src/od_training/cli/main.py`.
- Preferred implementation target: modules under `src/od_training/*`.

## 2. Compatibility Layer Contract

Legacy mirror modules exist in `src/`:
- `src/runtime_config.py`
- `src/cli_utils.py`
- `src/device_utils.py`
- `src/converters.py`

Rule:
- If behavior changes in active modules and compatibility behavior should remain equivalent, sync the mirrored module too.

## 3. Configuration Contract

Runtime config path:
- `config/local_config.json` (gitignored)

Resolution order (must remain stable unless intentionally changed):
1. CLI argument
2. local config file
3. environment variable

Current env vars:
- `ROBOFLOW_API_KEY`
- `ROBOFLOW_WORKSPACE`
- `ROBOFLOW_PROJECT`

## 4. Dataset Pipeline Contract

- Dataset manager entrypoint: `src/od_training/dataset/manager.py`.
- Default export mode is symlink-based zero-copy behavior.
- Portable export is opt-in via `--copy-images`.
- COCO sidecar export per split is expected where pipeline uses hybrid output.

## 5. CLI Contract

When adding/changing commands:
- Update dispatch map in `src/od_training/cli/main.py`.
- Keep `scripts/odt.py` wrapper behavior unchanged unless absolutely necessary.
- Update `docs/CLI.md` in same change set.

## 6. Testing Contract

- Use `pytest` markers from `pytest.ini`: `unit`, `integration`, `slow`, `gpu`.
- Add/adjust tests for behavior changes.
- Prefer fast, targeted tests first, then broader suite if needed.

## 7. Documentation Contract

- All classes, methods, and functions should keep concise, accurate docstrings.
- Update docs when contracts change:
  - CLI: `docs/CLI.md`
  - module behavior: `docs/src/*`
  - repo state/progress: `docs/dev/status.md`, `docs/dev/progress.md`
  - agent-specific standards/context: `docs/agent/*`

## 8. File-System Contract

These paths are runtime artifacts and are gitignored:
- `data/`, `runs/`, `output/`, `external/`, `config/local_config.json`

Implications:
- Do not rely on persistent tracked contents there.
- Scripts/tests may create transient artifacts in those locations.

`external/` specific rules:
- Purpose is local reference clones for reading upstream code/docs.
- Never import from `external/` in runtime code or tests.
- If using an external repo as behavioral evidence, record its commit SHA.
