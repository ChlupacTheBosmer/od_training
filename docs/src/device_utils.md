# Module: `src.device_utils`

- File: `src/device_utils.py`
- Canonical equivalent: `src.od_training.utility.device`

## Purpose

Resolves a compute device (`cpu` / `cuda`) with explicit override and environment-driven fallback.

## Functions

### `resolve_device(explicit_device=None)`

Resolution order:

1. explicit `explicit_device` argument
2. non-empty `CUDA_VISIBLE_DEVICES` -> `"cuda"`
3. `torch.cuda.is_available()` -> `"cuda"`
4. fallback -> `"cpu"`

Returns a device string consumed by training/inference modules.

## Notes

This module prints selection reasoning to stdout and is duplicated in `src.od_training.utility.device`.
