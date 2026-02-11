"""Inspect FiftyOne YOLO export path behavior using a tiny synthetic dataset."""

import fiftyone as fo
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime_config import ensure_local_config

ensure_local_config()

def test_export():
    """Export a one-sample dataset and print generated ``dataset.yaml``."""
    # Load dummy dataset
    if "test_export_ds" in fo.list_datasets():
        fo.delete_dataset("test_export_ds")
    
    dataset = fo.Dataset("test_export_ds")
    sample = fo.Sample(filepath="data/dummy_yolo/images/test_img.jpg")
    sample["ground_truth"] = fo.Detections(detections=[fo.Detection(label="test", bounding_box=[0.1, 0.1, 0.2, 0.2])])
    dataset.add_sample(sample)
    
    export_dir = "runs/test_export_yolo"
    
    # Export
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split="train"
    )
    
    # Check yaml
    yaml_path = os.path.join(export_dir, "dataset.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            print(f"--- {yaml_path} ---")
            print(f.read())
            print("----------------")
    else:
        print("dataset.yaml not found!")

if __name__ == "__main__":
    # Create dummy image content if needed (though FiftyOne might complain if file missing, 
    # but for export path check, we might need actual file presence for it to determine structure?)
    # Let's rely on the existing dummy_yolo data if present, or create a dummy file.
    os.makedirs("data/dummy_yolo/images", exist_ok=True)
    with open("data/dummy_yolo/images/test_img.jpg", "wb") as f:
        f.write(b"") # Empty file checks might fail, but let's try.
        
    test_export()
