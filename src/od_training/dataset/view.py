"""Dataset viewing helpers for FiftyOne-backed YOLO datasets."""

import argparse
import os

import fiftyone as fo

from ..utility.runtime_config import ensure_local_config

ensure_local_config()


def check_dataset_exists(name):
    """Return whether a FiftyOne dataset with ``name`` is available."""
    return name in fo.list_datasets()


def import_dataset(name: str, import_dir: str):
    """Import a YOLO dataset into FiftyOne.

    Args:
        name: Target FiftyOne dataset name.
        import_dir: Directory path or explicit YAML file path.

    Returns:
        Imported ``fiftyone.Dataset`` instance.
    """
    dataset_dir = os.path.abspath(import_dir)
    yaml_path = None
    if os.path.isfile(dataset_dir):
        yaml_path = dataset_dir
        dataset_dir = os.path.dirname(dataset_dir)

    kwargs = {}
    if yaml_path:
        kwargs["yaml_path"] = yaml_path

    return fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        name=name,
        **kwargs,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the dataset view command."""
    parser = argparse.ArgumentParser(description="View FiftyOne Dataset")
    parser.add_argument("name", help="Name of the FiftyOne dataset to view")
    parser.add_argument("--import-dir", help="Optional: Import YOLO dataset from this directory if missing.")
    return parser


def main(argv=None):
    """Open a dataset in FiftyOne, importing from disk when requested.

    Args:
        argv: Optional argument list. Uses ``sys.argv`` when omitted.

    Returns:
        Exit code ``0`` on success, ``1`` when dataset resolution fails.
    """
    args = build_parser().parse_args(argv)

    if check_dataset_exists(args.name):
        print(f"Loading existing dataset: {args.name}")
        dataset = fo.load_dataset(args.name)
    elif args.import_dir:
        print(f"Dataset {args.name} not found. Importing from {args.import_dir}...")
        dataset = import_dataset(args.name, args.import_dir)
    else:
        print(f"Error: Dataset '{args.name}' not found and no --import-dir specified.")
        print("Available datasets:", fo.list_datasets())
        return 1

    print("Launching App...")
    session = fo.launch_app(dataset)
    session.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
