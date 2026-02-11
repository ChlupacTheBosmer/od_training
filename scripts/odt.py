#!/usr/bin/env python3
"""Unified CLI wrapper."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.od_training.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
