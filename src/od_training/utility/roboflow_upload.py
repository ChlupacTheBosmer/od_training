"""Upload trained weights to Roboflow.

Supports API key via local config file (``config/local_config.json``) or an
explicit ``--api-key`` override. Pre-flight checks verify the weights file
exists before attempting to authenticate.
"""

import argparse
import logging
from pathlib import Path


from .runtime_config import (
    ensure_local_config,
    get_config_path,
    get_roboflow_api_key,
    get_roboflow_default,
)

ensure_local_config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_weights(
    api_key: str,
    workspace: str,
    project_name: str,
    version_num: int,
    weights_path: str,
    model_type: str = "yolov8",
):
    """Upload model weights to Roboflow.

    Args:
        api_key: Roboflow API key.
        workspace: Roboflow Workspace ID.
        project_name: Roboflow Project ID.
        version_num: Dataset version number.
        weights_path: Path to the weights file (e.g. ``.pt``).
        model_type: Model type string for Roboflow deploy
            (``"yolov8"``, ``"yolov5"``, ``"rf-detr"``, etc.).
    """
    # --- Preflight checks ---
    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if weights_file.stat().st_size == 0:
        raise ValueError(f"Weights file is empty (0 bytes): {weights_path}")

    logger.info("Weights file OK: %s (%.1f MB)", weights_path, weights_file.stat().st_size / 1e6)

    # --- Import roboflow lazily (heavy, not always installed) ---
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError(
            "roboflow package is not installed. "
            "Install it with: pip install roboflow"
        )

    logger.info("Authenticating with Roboflow…")
    rf = Roboflow(api_key=api_key)

    logger.info("Accessing %s/%s v%d…", workspace, project_name, version_num)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)

    logger.info("Uploading weights (type: %s)…", model_type)
    try:
        version.deploy(model_type=model_type, model_path=weights_path)
        logger.info("Upload successful ✓")
    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Build parser for Roboflow weight upload arguments."""
    parser = argparse.ArgumentParser(description="Upload weights to Roboflow.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Roboflow API Key. If omitted, value is loaded from local config.",
    )
    parser.add_argument(
        "--workspace",
        required=False,
        help="Roboflow Workspace ID (falls back to local config)",
    )
    parser.add_argument(
        "--project",
        required=False,
        help="Roboflow Project ID (falls back to local config)",
    )
    parser.add_argument("--version", type=int, required=True, help="Dataset Version Number")
    parser.add_argument("--weights", required=True, help="Path to weights file (.pt)")
    parser.add_argument(
        "--type",
        default="yolov8",
        help="Model type (yolov8, yolov5, rf-detr, etc.)",
    )
    return parser


def main(argv=None):
    """CLI entrypoint for Roboflow model upload.

    Args:
        argv: Optional argument list. Uses ``sys.argv`` when omitted.

    Returns:
        Exit code ``0`` on success.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        api_key = get_roboflow_api_key(args.api_key)
    except ValueError as e:
        parser.error(str(e))

    workspace = args.workspace or get_roboflow_default("workspace")
    project = args.project or get_roboflow_default("project")

    if not workspace:
        parser.error(
            "Workspace is required. Provide --workspace or set roboflow.workspace "
            f"in {get_config_path()}"
        )
    if not project:
        parser.error(
            "Project is required. Provide --project or set roboflow.project "
            f"in {get_config_path()}"
        )

    upload_weights(
        api_key,
        workspace,
        project,
        args.version,
        args.weights,
        args.type,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
