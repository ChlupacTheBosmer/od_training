import argparse
import sys
import os
import shutil
from pathlib import Path
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def train_rfdetr(dataset_dir: str, model_type: str, epochs: int, batch_size: int, lr: float, project_name: str, exp_name: str, **kwargs):
    """
    Trains RF-DETR on a local COCO dataset.
    """
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
        "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        "tensorboard": True, # Force enable native logging
        "wandb": True,       # Force enable native logging if installed
    }
    train_args.update(kwargs)
    
    logger.info(f"Train args: {train_args}")
    
    try:
        model.train(**train_args)
    except FileNotFoundError as e:
        if "checkpoint_best_regular.pth" in str(e):
            logger.warning("Training finished but failed to copy 'best' checkpoint (likely due to short run). Ignoring error.")
        else:
            raise e
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Post-training artifact management (RF-DETR saves to runs/{date} by default)
    # We might want to move/rename the best checkpoint to a predictable location if needed
    # logic here depends on exact rfdetr version output structure
    
    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RF-DETR models.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory (must contain train/valid/test subfolders with COCO json)")
    parser.add_argument("--model", type=str, default="rf-detr-resnet50", help="Model architecture (rf-detr-resnet50, rf-detr-resnet101)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--project", type=str, default="RF-DETR_Training")
    parser.add_argument("--name", type=str, default="exp")
    
    # Advanced args
    args, unknown = parser.parse_known_args()
    
    # Parse kwargs similar to YOLO script
    kwargs = {}
    i = 0
    while i < len(unknown):
        key = unknown[i]
        if key.startswith("--"):
            key = key[2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                kwargs[key] = unknown[i+1] # Keep as string/auto-convert inside rfdetr if needed
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            i += 1

    train_rfdetr(
        args.dataset, 
        args.model, 
        args.epochs, 
        args.batch, 
        args.lr, 
        args.project, 
        args.name, 
        **kwargs
    )
