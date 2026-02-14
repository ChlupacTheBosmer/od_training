"""od_training modular package.

Intentionally avoids eager submodule imports so package import stays lightweight
and does not force optional training/runtime dependencies at import time.
"""

__all__ = ["dataset", "train", "infer", "utility"]
