"""CLI wrapper for dataset format conversion helpers.

Delegates conversion work to ``od_training.dataset.convert`` while keeping a
small command interface for script and unified CLI usage.
"""

import argparse
import sys


from ..utility.runtime_config import ensure_local_config

ensure_local_config()

from .convert import convert_yolo_to_coco, convert_coco_to_yolo, validate_dataset

def main(argv=None):
    """Run YOLO<->COCO conversion from command-line arguments.

    Args:
        argv: Optional argument list. Uses ``sys.argv`` when omitted.

    Returns:
        Exit code ``0`` on success.
    """
    parser = argparse.ArgumentParser(description="Convert between YOLO and COCO formats.")
    parser.add_argument("mode", choices=["yolo2coco", "coco2yolo"], help="Conversion direction")
    parser.add_argument("--input", required=True, help="Input directory (YOLO labels dir or COCO json dir)")
    parser.add_argument("--images", required=False, help="Image directory (Required for YOLO->COCO)")
    parser.add_argument("--output", required=True, help="Output path (File for COCO JSON, Directory for YOLO)")
    
    args = parser.parse_args(argv)
    
    if args.mode == "yolo2coco":
        if not args.images:
            print("Error: --images directory is required for YOLO -> COCO conversion (to read dimensions).")
            sys.exit(1)
        convert_yolo_to_coco(args.input, args.images, args.output)
        
    elif args.mode == "coco2yolo":
        convert_coco_to_yolo(args.input, args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
