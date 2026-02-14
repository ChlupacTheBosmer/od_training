#!/usr/bin/env python3
"""Compatibility wrapper for the packaged ``odt`` entrypoint.

Preferred usage is the installed console script:
    odt <group> <command> [args...]

This wrapper intentionally avoids path injection. Install the repo in editable
mode first (`python -m pip install -e .`) and then invoke either `odt` or this
wrapper script.
"""

from od_training.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
