# Module: `src.od_training.utility.device`

- File: `src/od_training/utility/device.py`

## Purpose

Resolves compute device target for model operations.

## Functions

### `resolve_device(explicit_device=None)`

Resolution order:

1. explicit CLI value
2. `CUDA_VISIBLE_DEVICES`
3. `torch.cuda.is_available()` probe
4. CPU fallback

Returns device string.
