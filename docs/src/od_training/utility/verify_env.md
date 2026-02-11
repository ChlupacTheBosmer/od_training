# Module: `src.od_training.utility.verify_env`

- File: `src/od_training/utility/verify_env.py`
- CLI command: `odt utility verify-env`

## Purpose

Environment diagnostics for critical dependencies and accelerator availability.

## Functions

### `verify()`

Runs the full verification sequence.

Checks:

- Python version
- critical libs: `numpy`, `pydantic`, `torch`, `ultralytics`
- optional libs: `rfdetr`, `roboflow`
- hardware state via `_check_hardware`

Exit behavior:

- `sys.exit(1)` when any critical dependency is missing
- `sys.exit(0)` otherwise

### `_check_hardware()`

Prints CUDA and MPS availability using `torch`.

### `main(argv=None)`

Wrapper entrypoint for CLI compatibility. Calls `verify()` and returns `0` when execution continues.
