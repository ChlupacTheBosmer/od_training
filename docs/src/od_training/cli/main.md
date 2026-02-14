# Module: `src.od_training.cli.main`

- File: `src/od_training/cli/main.py`

## Purpose

Implements unified `odt` command dispatch. This module is intentionally thin and delegates to submodule `main(argv)` handlers.

Console script entrypoint:

- `odt` (defined in `pyproject.toml`)

## Globals

### `DISPATCH`

Mapping of `(group, command)` pairs to handler callables.

Supported commands:

- `dataset manage`
- `dataset convert`
- `dataset view`
- `dataset augment-preview`
- `train yolo`
- `train rfdetr`
- `infer run`
- `utility verify-env`
- `utility config-init`
- `utility config-show`
- `utility upload-weights`
- `utility download-roboflow`

## Functions

### `build_parser() -> argparse.ArgumentParser`

Builds the top-level parser with positional arguments:

- `group` (`dataset`, `train`, `infer`, `utility`)
- `command`
- `args` (remaining tokens forwarded to subcommand)

### `main(argv=None) -> int`

Parses command, resolves handler from `DISPATCH`, forwards trailing args, and returns handler exit code.

Behavior:

- Returns `2` for unknown command combinations.
- Strips a leading `--` in forwarded args when present.
