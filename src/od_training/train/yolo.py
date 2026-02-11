import argparse
import os
from ultralytics import YOLO
from clearml import Task
import logging


from ..utility.runtime_config import ensure_local_config
from ..utility.device import resolve_device
from ..utility.cli import parse_unknown_args

ensure_local_config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_yolo(model_name: str, data_yaml: str, project_name: str, exp_name: str, **kwargs):
    """
    Wrapper for Ultralytics YOLO training with ClearML integration.
    
    Args:
        model_name: Model version (e.g., 'yolo11n.pt', 'yolo26n.pt').
        data_yaml: Path to data.yaml.
        project_name: ClearML/YOLO project name.
        exp_name: Experiment name.
        **kwargs: Any additional arguments passed to model.train().
    """
    logger.info(f"Initializing ClearML Task: {project_name}/{exp_name}")
    # Initialize ClearML task
    # Note: Ultralytics has auto-clearml, but explicit init gives us more control over task naming
    try:
        task = Task.init(project_name=project_name, task_name=exp_name, output_uri=True)
    except Exception as e:
        logger.warning(f"ClearML initialization failed: {e}. Proceeding without ClearML logging.")
        task = None
    
    logger.info(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Auto-configure MuSGD for YOLO26 if not explicitly overridden
    if "yolo26" in model_name.lower() and "optimizer" not in kwargs:
        logger.info("Detected YOLO26: Setting default optimizer to 'MuSGD'")
        kwargs["optimizer"] = "MuSGD"

    logger.info(f"Starting training with args: {kwargs}")
    
    # Train
    results = model.train(
        data=data_yaml,
        project=project_name,
        name=exp_name,
        **kwargs
    )
    
    # Export to TensorRT if on GPU and not just a dry run
    if kwargs.get("device") != "cpu" and kwargs.get("epochs", 0) > 1:
        logger.info("Exporting to TensorRT engine...")
        try:
             # YOLO26 specific: end2end export
             end2end = "yolo26" in model_name.lower()
             model.export(format="engine", end2end=end2end)
        except Exception as e:
            logger.warning(f"Export failed: {e}")

    logger.info("Training complete.")
    return results

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO models (v11/v26) with full config support.")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Model weights (yolo11n.pt, yolo26n.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--project", type=str, default="YOLO_Training", help="Project name")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, 0, 0,1, etc.)")
    return parser


def main(argv=None) -> int:
    args, unknown = build_parser().parse_known_args(argv)
    kwargs = parse_unknown_args(unknown)
    train_args = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": resolve_device(args.device),
        **kwargs,
    }
    train_yolo(args.model, args.data, args.project, args.name, **train_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
