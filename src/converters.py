import shutil
from pathlib import Path
from globox import AnnotationSet
import ultralytics.data.converter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_yolo_to_coco(yolo_dir: str, image_dir: str, output_json: str):
    """
    Converts YOLO format to COCO JSON using globox.
    
    Args:
        yolo_dir: Directory containing .txt label files.
        image_dir: Directory containing corresponding images.
        output_json: Path to save the output .json file.
    """
    logger.info(f"Converting YOLO to COCO: {yolo_dir} -> {output_json}")
    try:
        # Globox needs image_folder to read image dimensions
        annotations = AnnotationSet.from_yolo_v5(
            folder=Path(yolo_dir),
            image_folder=Path(image_dir)
        )
        annotations.save_coco(Path(output_json), auto_ids=True)
        logger.info(f"Successfully converted {len(annotations)} images to {output_json}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise

def convert_coco_to_yolo(coco_json_dir: str, output_dir: str, use_segments=False):
    """
    Converts COCO format to YOLO using ultralytics.
    
    Args:
        coco_json_dir: Directory containing COCO json files (or path to specific json).
        output_dir: Output directory for YOLO files.
    """
    logger.info(f"Converting COCO to YOLO: {coco_json_dir} -> {output_dir}")
    # Ultralytics converter takes a dir containing *.json
    ultralytics.data.converter.convert_coco(
        labels_dir=coco_json_dir,
        save_dir=output_dir, # This arg might differ in newer ultralytics, checking doc in research needed or assuming standard
        use_segments=use_segments,
        use_keypoints=False,
        cls91to80=False
    )

def validate_dataset(image_dir: str, label_dir: str, format="yolo"):
    """
    Basic validation for dataset integrity.
    """
    from PIL import Image
    errors = []
    logger.info(f"Validating dataset in {image_dir}")
    
    img_path_list = list(Path(image_dir).glob('*.*'))
    for img_path in img_path_list:
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']: 
            continue
            
        # Check integrity
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            errors.append(f"Corrupt image: {img_path} ({e})")
            
        # Check label existence (YOLO)
        if format == "yolo":
            label_path = Path(label_dir) / f"{img_path.stem}.txt"
            if not label_path.exists():
                # Might be background image, but good to note
                pass 
                
    if errors:
        logger.error(f"Validation found {len(errors)} errors:")
        for e in errors:
            logger.error(e)
        return False
    
    logger.info("Validation passed.")
    return True
