import sys
from pathlib import Path


from ..utility.runtime_config import ensure_local_config

ensure_local_config()

import fiftyone as fo
import fiftyone.zoo as foz
import albumentations as A
import argparse
import os
import cv2
import numpy as np

def define_pipeline(width=640, height=640):
    """
    Define the Albumentations pipeline.
    Customize this function to add/remove augmentations.
    """
    transform = A.Compose([
        A.RandomCrop(width=width, height=height, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ToGray(p=0.1),
        # Add more augmentations here
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    return transform

def augment_and_view(dataset_dir: str, name: str = "my_dataset", yaml_path: str = None):
    """
    Load a YOLO dataset into FiftyOne, apply augmentations via plugin (if available) 
    or manually, and launch the App.
    """
    # 1. Load Dataset
    dataset_dir = os.path.abspath(dataset_dir)
    print(f"Loading dataset from {dataset_dir}...")
    
    if name in fo.list_datasets():
        dataset = fo.load_dataset(name)
    else:
        # Determine YAML path if not provided
        if yaml_path is None:
             possible_yaml = os.path.join(dataset_dir, "data.yaml")
             if os.path.exists(possible_yaml):
                 yaml_path = possible_yaml
        
        load_kwargs = {}
        if yaml_path:
            load_kwargs["yaml_path"] = os.path.abspath(yaml_path)

        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            name=name,
            **load_kwargs
        )

    print(f"Dataset '{name}' loaded with {len(dataset)} samples.")

    # 2. Check for Albumentations plugin
    # Note: FiftyOne plugins are typically used in the App context.
    # Programmatically, we can verify the pipeline logic by generating a few samples.
    
    print("Generating augmented samples for verification...")
    transform = define_pipeline()
    
    # Create a view for augmentation visualization
    # In a real workflow, you might want to materialize these augmentations
    # but for "planning" and "tuning", visualization is key.
    
    # Since we can't easily script the "App interactions", we will:
    # 1. Print instructions to use the plugin in the App.
    # 2. Run a small manual test on one sample and save it to disk for user check.
    
    if len(dataset) == 0:
        print("Dataset is empty. Skipping verification generation.")
    else:
        sample = dataset.first()
        img_path = sample.filepath
        image = cv2.imread(img_path)
        if image is None:
             print(f"Could not read image: {img_path}")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract bboxes (FiftyOne stores as [top-left-x, top-left-y, w, h] normalized)
            bboxes = []
            class_labels = []
            h, w = image.shape[:2]
            
            if sample.ground_truth:
                for det in sample.ground_truth.detections:
                    # FiftyOne: [x, y, w, h] normalized
                    # Albumentations YOLO: [x_center, y_center, w, h] normalized? 
                    # Wait, Albumentations 'yolo' format is [x_center, y_center, w, h] normalized.
                    # FiftyOne is [x_min, y_min, w, h] normalized.
                    
                    x, y, w_box, h_box = det.bounding_box
                    # Convert to YOLO format for Albumentations
                    xc = x + w_box / 2
                    yc = y + h_box / 2
                    bboxes.append([xc, yc, w_box, h_box])
                    class_labels.append(det.label)
                    
            # Apply
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                
                # Save verification image
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                
                # Draw boxes
                for bbox, label in zip(aug_bboxes, augmented['class_labels']):
                    xc, yc, wb, hb = bbox
                    h_aug, w_aug = aug_img.shape[:2]
                    
                    x1 = int((xc - wb / 2) * w_aug)
                    y1 = int((yc - hb / 2) * h_aug)
                    x2 = int((xc + wb / 2) * w_aug)
                    y2 = int((yc + hb / 2) * h_aug)
                    
                    cv2.rectangle(aug_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(aug_img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                os.makedirs("runs/augmentation_test", exist_ok=True)
                cv2.imwrite("runs/augmentation_test/aug_sample.jpg", aug_img_bgr)
                print("Saved augmented sample to runs/augmentation_test/aug_sample.jpg")
                
            except Exception as e:
                print(f"Augmentation failed on sample: {e}")

    # 3. Launch App instruction
    print("\n" + "="*50)
    print("TO VISUALIZE IN FIFTYONE APP:")
    print(f"1. Run: fiftyone app launch {name}")
    print("2. Open the browser.")
    print("3. Use the 'Compute' panel -> 'albumentations' to experiment interactively.")
    print("="*50 + "\n")
    
    # 4. Session (optional, blocks script)
    # session = fo.launch_app(dataset)
    # session.wait()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Augment data with FiftyOne and Albumentations")
    parser.add_argument("--dataset", type=str, required=True, help="Path to YOLO dataset directory (containing dataset.yaml not supported directly by FiftyOne YOLO importer usually, needs dir structure)")
    parser.add_argument("--name", type=str, default="my_dataset", help="FiftyOne dataset name")
    parser.add_argument("--yaml", type=str, help="Path to data.yaml if not dataset.yaml")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    augment_and_view(args.dataset, args.name, args.yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
