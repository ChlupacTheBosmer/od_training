# Module: `src.od_training.utility.runtime_config`

- File: `src/od_training/utility/runtime_config.py`

## Purpose

Local runtime config management for credentials and machine-specific defaults.
Config content is parsed through typed Pydantic models.

## Typed Models

### `RoboflowRuntimeConfig`

Typed `roboflow` section (`api_key`, `workspace`, `project`).

### `LocalRuntimeConfig`

Typed root model for local config (`_instructions`, `roboflow`, `data_dir`).

## Config Resolution

Default resolution order:

1. `ODT_CONFIG_PATH` environment variable
2. repo-local `config/local_config.json` (if present)
3. `~/.config/od_training/local_config.json` (or `$XDG_CONFIG_HOME/od_training/local_config.json`)

## Functions

### `get_repo_root() -> Path`

Returns repository root (`parents[3]` from this module path) for editable-source execution.

### `resolve_default_local_config_path() -> Path`

Resolves portable default config location using the priority above.

### `get_config_path(config_path: Path | None = None) -> Path`

Returns explicit config path when provided, otherwise uses the default resolver.

### `ensure_local_config(config_path: Path | None = None) -> tuple[Path, bool]`

Creates local config template when missing.

### `load_local_config(config_path: Path | None = None) -> dict[str, Any]`

Loads and validates config object.

### `parse_local_config(raw: dict[str, Any]) -> LocalRuntimeConfig`

Parses raw config dict into typed runtime model and raises `ValueError` for
schema violations.

### `_is_placeholder(value: Any) -> bool`

Detects placeholder-like values (`<...>` or blank).

### `get_roboflow_api_key(explicit_key: str | None = None, config_path: Path | None = None) -> str`

Resolves API key using explicit -> typed config -> environment priority.

### `get_roboflow_default(field: str, config_path: Path | None = None) -> str | None`

Resolves optional `workspace`/`project` values using typed config ->
environment priority.

### `get_data_dir(config_path: Path | None = None) -> str`

Returns configured data root, defaulting to `data`.
