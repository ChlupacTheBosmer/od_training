"""Shared device detection and selection utilities.

Provides a unified ``resolve_device()`` used by both YOLO and RF-DETR training
scripts so that CUDA detection is consistent and overridable via ``--device``.
"""

import os


def resolve_device(explicit_device=None):
    """Determine the best available compute device.

    Resolution priority:
        1. Explicit ``explicit_device`` argument (e.g. ``"cuda:0"``, ``"cpu"``).
        2. ``CUDA_VISIBLE_DEVICES`` environment variable — if set, assume CUDA.
        3. ``torch.cuda.is_available()`` probe.
        4. Fallback to ``"cpu"``.

    Args:
        explicit_device: Optional device string provided via CLI ``--device``.

    Returns:
        str: Device identifier suitable for passing to training frameworks
            (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
    """
    # 1. Explicit override always wins
    if explicit_device is not None:
        print(f"[device] Using explicit device: {explicit_device}")
        return explicit_device

    # 2. Environment variable hint
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env is not None and cuda_env.strip() != "":
        print(f"[device] CUDA_VISIBLE_DEVICES={cuda_env} → using cuda")
        return "cuda"

    # 3. Torch probe (lazy import — torch may not be on PATH in all contexts)
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[device] CUDA available ({gpu_name}) → using cuda")
            return "cuda"
    except ImportError:
        pass

    print("[device] No GPU detected → using cpu")
    return "cpu"
