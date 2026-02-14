"""RF-DETR training entrypoints and CLI helpers."""

import argparse
from pathlib import Path
import os
from rfdetr import (
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRLarge,
    RFDETRXLarge,
    RFDETR2XLarge,
    RFDETRBase
)
from clearml import Task
import logging


from ..utility.device import resolve_device
from ..utility.cli import parse_unknown_args
from .preflight import validate_rfdetr_training_inputs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_MAP = {
    "rfdetr_nano": RFDETRNano,
    "rfdetr_small": RFDETRSmall,
    "rfdetr_medium": RFDETRMedium,
    "rfdetr_large": RFDETRLarge,
    "rfdetr_xlarge": RFDETRXLarge,
    "rfdetr_2xlarge": RFDETR2XLarge,
    "rf-detr-resnet50": RFDETRMedium, # Mapping ResNet50 to Medium as a sensible default
    "rf-detr-base": RFDETRBase
}

def train_rfdetr(
    dataset_dir: str,
    model_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    project_name: str,
    exp_name: str,
    device: str = None,
    validate_data: bool = False,
    fail_on_validation_warnings: bool = False,
    **kwargs,
):
    """Train RF-DETR on a local dataset with optional pass-through kwargs.

    This function is the RF-DETR counterpart to ``train.yolo.train_yolo`` and
    is intended to be called from the unified CLI dispatcher.

    Args:
        dataset_dir: Dataset root directory expected by RF-DETR.
        model_type: RF-DETR model key or alias.
        epochs: Number of training epochs.
        batch_size: Batch size per step.
        lr: Learning rate forwarded as ``learning_rate``.
        project_name: Tracking/output project name.
        exp_name: Run name for tracking/output grouping.
        device: Optional explicit device override.
        validate_data: If true, run dataset preflight validation first.
        fail_on_validation_warnings: If true, warnings fail preflight.
        **kwargs: Additional RF-DETR ``model.train`` arguments.
    """
    if validate_data:
        preflight = validate_rfdetr_training_inputs(
            dataset_dir=dataset_dir,
            fail_on_warnings=fail_on_validation_warnings,
        )
        logger.info(
            "RF-DETR preflight OK: splits=%s warnings=%d",
            sorted(preflight.get("splits", {}).keys()),
            len(preflight.get("warnings", [])),
        )

    logger.info(f"Initializing ClearML Task: {project_name}/{exp_name}")
    try:
        task = Task.init(project_name=project_name, task_name=exp_name, output_uri=True)
    except Exception as e:
        logger.warning(f"ClearML initialization failed: {e}. Proceeding without ClearML logging.")
        task = None

    logger.info(f"Loading RF-DETR model: {model_type}")
    
    # Resolve model class
    model_cls = MODEL_MAP.get(model_type.lower())
    if model_cls is None:
        # Fallback: try to find class by name insensitive
        for k, v in MODEL_MAP.items():
            if k in model_type.lower():
                model_cls = v
                break
    
    if model_cls is None:
         logger.warning(f"Model type '{model_type}' not found in map. Defaulting to RFDETRMedium.")
         model_cls = RFDETRMedium

    model = model_cls()
    
    # Ensure dataset exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    logger.info(f"Starting training on {dataset_dir} for {epochs} epochs...")
    
    # Construct arguments for model.train()
    # We pass explicit args and then any extra kwargs allow for future API changes or advanced configs
    train_args = {
        "dataset_dir": dataset_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "device": resolve_device(device),
        "tensorboard": True,
        "wandb": True,
    }
    train_args.update(kwargs)
    
    logger.info(f"Train args: {train_args}")
    
    try:
        model.train(**train_args)
    except FileNotFoundError as e:
        if "checkpoint_best_regular.pth" in str(e):
            logger.warning(
                "Training finished but failed to copy 'best' checkpoint "
                "(likely due to short run). Checking for other checkpoints..."
            )
            # Verify that at least one checkpoint was actually produced
            output_dir = train_args.get("output_dir", ".")
            checkpoints = list(Path(output_dir).rglob("*.pth"))
            if not checkpoints:
                logger.error(
                    "No .pth checkpoints found in output directory '%s'. "
                    "Training may have failed entirely.",
                    output_dir,
                )
                raise
            logger.info(
                "Found %d checkpoint(s): %s",
                len(checkpoints),
                [str(p.name) for p in checkpoints],
            )
        else:
            raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Post-training artifact management (RF-DETR saves to runs/{date} by default)
    # We might want to move/rename the best checkpoint to a predictable location if needed
    # logic here depends on exact rfdetr version output structure
    
    logger.info("Training complete.")

def build_parser() -> argparse.ArgumentParser:
    """Build parser for RF-DETR training CLI arguments."""
    parser = argparse.ArgumentParser(description="Train RF-DETR models.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory (must contain train/valid/test subfolders with COCO json)")
    parser.add_argument("--model", type=str, default="rf-detr-resnet50", help="Model architecture (rf-detr-resnet50, rf-detr-resnet101)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--project", type=str, default="RF-DETR_Training")
    parser.add_argument("--name", type=str, default="exp")
    
    parser.add_argument("--device", type=str, default=None, help="Device to train on (e.g. cuda, cuda:0, cpu). Auto-detected if not set.")
    parser.add_argument(
        "--no-validate-data",
        action="store_true",
        help="Skip dataset preflight validation before training.",
    )
    parser.add_argument(
        "--fail-on-validation-warnings",
        action="store_true",
        help="Treat dataset preflight warnings as errors.",
    )
    return parser


def main(argv=None) -> int:
    """CLI entrypoint for RF-DETR training.

    Args:
        argv: Optional argument list. Uses ``sys.argv`` when omitted.

    Returns:
        Exit code ``0`` on success.
    """
    args, unknown = build_parser().parse_known_args(argv)
    kwargs = parse_unknown_args(unknown)

    train_rfdetr(
        args.dataset,
        args.model,
        args.epochs,
        args.batch,
        args.lr,
        args.project,
        args.name,
        device=args.device,
        validate_data=not args.no_validate_data,
        fail_on_validation_warnings=args.fail_on_validation_warnings,
        **kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
