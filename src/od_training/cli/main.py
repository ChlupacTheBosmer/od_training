"""Unified CLI dispatch for the modular package.

This module is intentionally thin and delegates to submodule `main(argv)`
functions to keep CLI entrypoints separate from core module logic.
"""

from __future__ import annotations

import argparse

from ..dataset import augment_preview, convert_cli, manager, view
from ..infer import runner
from ..train import rfdetr, yolo
from ..utility import roboflow_download, roboflow_upload, verify_env


DISPATCH = {
    ("dataset", "manage"): manager.main,
    ("dataset", "convert"): convert_cli.main,
    ("dataset", "view"): view.main,
    ("dataset", "augment-preview"): augment_preview.main,
    ("train", "yolo"): yolo.main,
    ("train", "rfdetr"): rfdetr.main,
    ("infer", "run"): runner.main,
    ("utility", "verify-env"): verify_env.main,
    ("utility", "upload-weights"): roboflow_upload.main,
    ("utility", "download-roboflow"): roboflow_download.main,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="od_training unified CLI",
        epilog=(
            "Examples:\n"
            "  odt dataset manage --dataset-dir data/dummy_yolo/data.yaml --name demo --split\n"
            "  odt train yolo --data runs/export/dataset.yaml --epochs 10 --batch 8\n"
            "  odt infer run --type yolo --model yolo11n.pt --source data/images --save-dir runs/infer\n"
            "  odt utility verify-env"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("group", choices=["dataset", "train", "infer", "utility"])
    parser.add_argument("command")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv=None) -> int:
    ns = build_parser().parse_args(argv)
    key = (ns.group, ns.command)
    entry = DISPATCH.get(key)
    if entry is None:
        available = ", ".join(f"{g} {c}" for g, c in sorted(DISPATCH))
        print(f"Unknown command: {ns.group} {ns.command}")
        print(f"Available commands: {available}")
        return 2

    forwarded = list(ns.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return int(entry(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())
