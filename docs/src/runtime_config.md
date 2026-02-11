# Module: `src.runtime_config`

- File: `src/runtime_config.py`
- Canonical equivalent: `src.od_training.utility.runtime_config`

## Purpose

Creates and reads local runtime configuration (`config/local_config.json`) and resolves Roboflow credentials/defaults.

## Configuration keys

- `roboflow.api_key`
- `roboflow.workspace`
- `roboflow.project`
- `data_dir`

## Functions

### `get_repo_root() -> Path`

Returns repository root path (`Path(__file__).resolve().parent.parent`).

### `get_config_path() -> Path`

Returns default local config path: `config/local_config.json` under repo root.

### `ensure_local_config(config_path: Path | None = None) -> tuple[Path, bool]`

Ensures config file exists.

- Creates template with placeholders when missing.
- Returns `(path, created_flag)`.

### `load_local_config(config_path: Path | None = None) -> dict[str, Any]`

Loads and validates config JSON.

- Raises `ValueError` for invalid JSON or non-object root.

### `_is_placeholder(value: Any) -> bool`

Returns `True` for empty/placeholder-like values (for example `<...>`).

### `get_roboflow_api_key(explicit_key: str | None = None) -> str`

Resolves API key by priority:

1. explicit argument
2. `roboflow.api_key` in local config
3. `ROBOFLOW_API_KEY` environment variable

Raises `ValueError` when unresolved.

### `get_roboflow_default(field: str) -> str | None`

Resolves optional `workspace` or `project` value by priority:

1. local config
2. environment variable (`ROBOFLOW_WORKSPACE` / `ROBOFLOW_PROJECT`)

Raises `ValueError` for unsupported fields.

### `get_data_dir() -> str`

Returns configured `data_dir` or fallback `"data"`.

## Notes

Prefer importing from `src.od_training.utility.runtime_config` in modular code.
