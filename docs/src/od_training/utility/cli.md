# Module: `src.od_training.utility.cli`

- File: `src/od_training/utility/cli.py`

## Purpose

Parses unknown `argparse` tokens into typed keyword arguments for downstream framework APIs.

## Functions

### `parse_unknown_args(unknown)`

Converts unknown CLI tokens to dictionary form with key normalization and type conversion.

### `_convert_value(value)`

Internal string-to-type converter used by `parse_unknown_args`.

Conversion order:

- booleans
- `None`
- comma-separated tuples
- integers
- floats
- raw string fallback
