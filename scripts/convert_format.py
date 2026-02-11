import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime_config import ensure_local_config

ensure_local_config()

from src.converters import convert_yolo_to_coco, convert_coco_to_yolo, validate_dataset

def main():
    parser = argparse.ArgumentParser(description="Convert between YOLO and COCO formats.")
    parser.add_argument("mode", choices=["yolo2coco", "coco2yolo"], help="Conversion direction")
    parser.add_argument("--input", required=True, help="Input directory (YOLO labels dir or COCO json dir)")
    parser.add_argument("--images", required=False, help="Image directory (Required for YOLO->COCO)")
    parser.add_argument("--output", required=True, help="Output path (File for COCO JSON, Directory for YOLO)")
    
    args = parser.parse_args()
    
    if args.mode == "yolo2coco":
        if not args.images:
            print("Error: --images directory is required for YOLO -> COCO conversion (to read dimensions).")
            sys.exit(1)
        convert_yolo_to_coco(args.input, args.images, args.output)
        
    elif args.mode == "coco2yolo":
        convert_coco_to_yolo(args.input, args.output)

if __name__ == "__main__":
    main()
