"""Format conversion utilities (YOLO ↔ COCO) and dataset validation.

Provides:
    - ``convert_yolo_to_coco()``: YOLO .txt → COCO JSON via *globox*.
    - ``convert_coco_to_yolo()``: COCO JSON → YOLO .txt, with *ultralytics*
      as the primary backend and *globox* as a fallback when ultralytics'
      internal API is unavailable.
    - ``validate_dataset()``: Checks image integrity, label existence, bbox
      validity, and column count.  Returns a ``ValidationReport``.
"""

import logging
from pathlib import Path
from typing import List, NamedTuple, Tuple

from globox import AnnotationSet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------
class ValidationReport(NamedTuple):
    """Container for dataset validation results."""

    ok: bool
    errors: List[Tuple[str, str]]    # [(path_or_context, message), ...]
    warnings: List[Tuple[str, str]]  # [(path_or_context, message), ...]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def convert_yolo_to_coco(yolo_dir: str, image_dir: str, output_json: str):
    """Convert YOLO format labels to COCO JSON using *globox*.

    Args:
        yolo_dir: Directory containing ``.txt`` label files.
        image_dir: Directory containing corresponding images (needed for
            image dimensions).
        output_json: Path to save the output ``.json`` file.
    """
    logger.info("Converting YOLO → COCO: %s → %s", yolo_dir, output_json)
    try:
        annotations = AnnotationSet.from_yolo_v5(
            folder=Path(yolo_dir),
            image_folder=Path(image_dir),
        )
        annotations.save_coco(Path(output_json), auto_ids=True)
        logger.info(
            "Successfully converted %d images to %s",
            len(annotations),
            output_json,
        )
    except Exception as e:
        logger.error("YOLO → COCO conversion failed: %s", e)
        raise


def convert_coco_to_yolo(
    coco_json_dir: str,
    output_dir: str,
    use_segments: bool = False,
):
    """Convert COCO JSON to YOLO format.

    Tries the *ultralytics* internal converter first.  If that API is
    unavailable (e.g. a breaking change in a new ultralytics release),
    falls back transparently to *globox*.

    Args:
        coco_json_dir: Directory containing COCO ``*.json`` files (or path
            to a specific JSON).
        output_dir: Output directory for YOLO ``.txt`` files.
        use_segments: If ``True``, export segmentation masks instead of
            bounding boxes.
    """
    logger.info("Converting COCO → YOLO: %s → %s", coco_json_dir, output_dir)

    # Primary: ultralytics (fast, well-tested)
    try:
        from ultralytics.data.converter import convert_coco

        convert_coco(
            labels_dir=coco_json_dir,
            save_dir=output_dir,
            use_segments=use_segments,
            use_keypoints=False,
            cls91to80=False,
        )
        return
    except (ImportError, AttributeError, TypeError) as e:
        logger.warning(
            "Ultralytics COCO converter unavailable (%s). "
            "Falling back to globox.",
            e,
        )

    # Fallback: globox
    json_dir = Path(coco_json_dir)
    json_files = list(json_dir.glob("*.json")) if json_dir.is_dir() else [json_dir]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for jf in json_files:
        annotations = AnnotationSet.from_coco(jf)
        annotations.save_yolo_v5(
            save_dir=out,
            label_to_id=None,  # auto-assign
        )
    logger.info("Fallback conversion complete via globox.")


# ---------------------------------------------------------------------------
# Dataset Validation
# ---------------------------------------------------------------------------
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def validate_dataset(
    image_dir: str,
    label_dir: str,
    format: str = "yolo",
    class_names: list = None,
):
    """Validate dataset integrity and return a ``ValidationReport``.

    Checks performed:
        1. Image file integrity (``PIL.Image.verify``).
        2. Matching label file existence for each image.
        3. Label column count (accepts 5 or 6 columns).
        4. Bounding box values in ``[0, 1]`` range (YOLO only).
        5. Class IDs within bounds (if ``class_names`` is provided).

    Args:
        image_dir: Path to the directory of images.
        label_dir: Path to the directory of YOLO ``.txt`` labels.
        format: Label format — currently only ``"yolo"`` is validated.
        class_names: Optional list of class names. If provided, class IDs
            are checked for validity.

    Returns:
        A ``ValidationReport`` with ``ok``, ``errors``, and ``warnings``.
    """
    from PIL import Image

    errors: List[Tuple[str, str]] = []
    warnings: List[Tuple[str, str]] = []
    img_path_list = [
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in _IMG_EXTS
    ]

    logger.info("Validating %d images in %s …", len(img_path_list), image_dir)

    for img_path in img_path_list:
        # --- Image integrity ---
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            errors.append((str(img_path), f"Corrupt image: {e}"))
            continue

        if format != "yolo":
            continue

        # --- Label file existence ---
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            warnings.append(
                (str(img_path), "No matching label file (background image?)")
            )
            continue

        # --- Label content validation ---
        _validate_yolo_label(label_path, class_names, errors, warnings)

    # --- Summary ---
    if errors:
        logger.error("Validation found %d error(s).", len(errors))
    if warnings:
        logger.warning("Validation found %d warning(s).", len(warnings))
    if not errors and not warnings:
        logger.info("Validation passed ✓")

    return ValidationReport(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _validate_yolo_label(label_path, class_names, errors, warnings):
    """Validate a single YOLO ``.txt`` label file."""
    path_str = str(label_path)

    with open(label_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue

            # Column count: 5 (standard) or 6 (with confidence) are valid
            if len(parts) not in (5, 6):
                errors.append(
                    (path_str, f"Line {line_no}: expected 5 or 6 columns, got {len(parts)}")
                )
                continue

            if len(parts) == 6:
                warnings.append(
                    (path_str, f"Line {line_no}: 6-column label (has confidence score)")
                )

            # Parse values
            try:
                cls_id = int(parts[0])
                coords = [float(v) for v in parts[1:5]]
            except ValueError:
                errors.append(
                    (path_str, f"Line {line_no}: non-numeric value in label")
                )
                continue

            # Bbox range check [0, 1]
            for i, val in enumerate(coords):
                if not (0.0 <= val <= 1.0):
                    errors.append(
                        (path_str,
                         f"Line {line_no}: coordinate [{i}]={val} out of [0,1] range")
                    )

            # Class ID bounds check
            if class_names is not None and cls_id >= len(class_names):
                errors.append(
                    (path_str,
                     f"Line {line_no}: class_id={cls_id} out of range "
                     f"(max={len(class_names) - 1})")
                )
