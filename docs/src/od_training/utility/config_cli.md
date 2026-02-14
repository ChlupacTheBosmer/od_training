# Module: `src.od_training.utility.config_cli`

- File: `src/od_training/utility/config_cli.py`
- CLI commands:
  - `odt utility config-init`
  - `odt utility config-show`

## Purpose

Provides explicit runtime-config management commands for creating and
inspecting local config JSON without relying on import-time side effects.

## Commands

### `config-init`

Creates config template if missing.

Args:

- `--config <path>` optional explicit local config path

Output:

- JSON payload with `ok`, `path`, and `state` (`created` or `exists`)

### `config-show`

Prints resolved local config as JSON.

Args:

- `--config <path>` optional explicit local config path
- `--mask-secrets` redacts secret-like keys (`api_key`, `token`, `secret`, `password`)

