"""Environment verification script.

Checks that all critical dependencies are importable and prints version
information.  Exits with code 1 if any *hard* requirement is missing.
"""

import sys


def verify():
    """Run environment checks and report results."""
    print("--- Environment Verification ---")
    print(f"Python: {sys.version.split()[0]}")

    hard_failures = 0

    # --- Critical packages ---
    # Numpy
    try:
        import numpy
        print(f"  Numpy: {numpy.__version__}")
    except ImportError as e:
        print(f"  FAIL: Numpy not found: {e}")
        hard_failures += 1

    # Pydantic
    try:
        import pydantic
        version = getattr(pydantic, "VERSION", None) or getattr(pydantic, "__version__", "unknown")
        print(f"  Pydantic: {version}")
    except ImportError:
        print("  FAIL: Pydantic not found")
        hard_failures += 1

    # Torch
    try:
        import torch
        print(f"  Torch: {torch.__version__}")
    except ImportError as e:
        print(f"  FAIL: Torch not found: {e}")
        hard_failures += 1

    # --- Model packages ---
    # Ultralytics
    try:
        import ultralytics
        print(f"  Ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        print(f"  FAIL: Ultralytics import failed: {e}")
        hard_failures += 1

    # RF-DETR (optional — won't cause exit(1))
    try:
        import rfdetr
        print("  RF-DETR: OK")
    except ImportError as e:
        print(f"  WARN: RF-DETR import failed: {e}")

    # Roboflow (optional)
    try:
        import roboflow
        print("  Roboflow: OK")
    except ImportError:
        print("  WARN: Roboflow import failed")

    # --- Hardware ---
    _check_hardware()

    # --- Result ---
    print("--- Verification Complete ---")
    if hard_failures:
        print(f"\n{hard_failures} critical package(s) missing. Fix before proceeding.")
        sys.exit(1)
    else:
        print("\nAll critical packages OK.")
        sys.exit(0)


def _check_hardware():
    """Detect available accelerators."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            print(f"  CUDA: available ({gpu})")
        else:
            print("  CUDA: not available")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  MPS: available (Metal Performance Shaders)")
        else:
            print("  MPS: not available")
    except ImportError:
        pass


if __name__ == "__main__":
    verify()
