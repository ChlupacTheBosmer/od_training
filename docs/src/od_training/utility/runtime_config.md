# Module: `src.od_training.utility.runtime_config`

- File: `src/od_training/utility/runtime_config.py`

## Purpose

Local runtime config management for credentials and machine-specific defaults.

## Functions

### `get_repo_root() -> Path`

Returns repository root (`parents[3]` from this module path).

### `get_config_path() -> Path`

Returns `config/local_config.json` path under repo root.

### `ensure_local_config(config_path: Path | None = None) -> tuple[Path, bool]`

Creates local config template when missing.

### `load_local_config(config_path: Path | None = None) -> dict[str, Any]`

Loads and validates config object.

### `_is_placeholder(value: Any) -> bool`

Detects placeholder-like values (`<...>` or blank).

### `get_roboflow_api_key(explicit_key: str | None = None) -> str`

Resolves API key using explicit -> config -> environment priority.

### `get_roboflow_default(field: str) -> str | None`

Resolves optional `workspace`/`project` values using config -> environment priority.

### `get_data_dir() -> str`

Returns configured data root, defaulting to `data`.
