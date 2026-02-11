# Module: `src.cli_utils`

- File: `src/cli_utils.py`
- Canonical equivalent: `src.od_training.utility.cli`

## Purpose

Parses unknown CLI arguments from `argparse.parse_known_args()` into typed keyword arguments for downstream trainer APIs.

## Functions

### `parse_unknown_args(unknown)`

Converts leftover CLI tokens (for example `--lr0 0.01 --freeze 10`) into a typed `dict`.

- Input: list of string tokens.
- Output: dictionary suitable for `**kwargs` forwarding.
- Behavior:
  - Bare flags become `True`.
  - Keys are normalized from kebab-case to snake_case.
  - Values are converted by `_convert_value`.

### `_convert_value(value)`

Converts a single string value using ordered coercion rules:

1. `"true"/"false"` -> `bool`
2. `"none"` -> `None`
3. comma-separated string -> `tuple` (recursively converted)
4. integer parse
5. float parse
6. raw string fallback

## Notes

This module is functionally duplicated in `src.od_training.utility.cli` and kept for compatibility with top-level imports.
