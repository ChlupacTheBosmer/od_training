"""Training module for YOLO and RF-DETR."""

from .rfdetr import train_rfdetr
from .yolo import train_yolo

__all__ = ["train_yolo", "train_rfdetr"]
