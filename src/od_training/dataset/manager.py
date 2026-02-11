import argparse
import json
from pathlib import Path
import fiftyone as fo
import fiftyone.utils.random as four
import albumentations as A
import cv2
import os
from collections import Counter
from tqdm import tqdm
import warnings


from ..utility.runtime_config import ensure_local_config

ensure_local_config()

# Suppress Albumentations version check warning (SSL/Network issues)
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

# =============================================================================
# AUGMENTATION PIPELINES
# =============================================================================
def get_augmentation_pipeline(width=640, height=640):
    """
    Define the Albumentations pipeline.
    Adjust this function to change your augmentation strategy.
    """
    transform = A.Compose([
        A.RandomCrop(width=width, height=height, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ToGray(p=0.1),
        # Ensure we resize to target at the end if crop didn't happen or was smaller?
        # A.Resize(width, height) # Optional, depends on consistency needs
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    return transform

# =============================================================================
# DATASET LOADING & SPLITTING
# =============================================================================
def load_or_create_dataset(dataset_dir, name, split_ratios=None, train_tag="train", val_tag="val", test_tag="test"):
    """
    Load a dataset from disk or FiftyOne DB. 
    If not in DB, imports from `dataset_dir` (YOLO format).
    If `split_ratios` provided, performs random split on untagged samples.
    """
    if name in fo.list_datasets():
        print(f"Loading existing dataset: {name}")
        dataset = fo.load_dataset(name)
    else:
        dataset_dir = os.path.abspath(dataset_dir)
        yaml_path = None
        
        # If user passed a file (e.g. data.yaml) instead of dir
        if os.path.isfile(dataset_dir):
            yaml_path = dataset_dir
            dataset_dir = os.path.dirname(dataset_dir)
            print(f"Inferred dataset directory from file: {dataset_dir}")
            
        print(f"Importing dataset from {dataset_dir}...")
        
        # Check for data.yaml if dataset.yaml doesn't exist and we don't have one yet
        if not yaml_path and not os.path.exists(os.path.join(dataset_dir, "dataset.yaml")):
             possible = os.path.join(dataset_dir, "data.yaml")
             if os.path.exists(possible):
                 yaml_path = possible
        
        kwargs = {}
        if yaml_path:
            kwargs["yaml_path"] = yaml_path

        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            name=name,
            **kwargs
        )
    
    # Perform Split if requested and tags are missing
    if split_ratios:
        # Check if we already have splits
        counts = dataset.count_sample_tags()
        # Check against the USER provided tag names
        if not any(x in counts for x in [train_tag, val_tag, test_tag]):
            print(f"Splitting dataset: {split_ratios} using tags {[train_tag, val_tag, test_tag]}")
            # four.random_split uses "train", "test", "val" keys in dictionary implies tags
            # We construct map:
            four.random_split(dataset, {train_tag: split_ratios.get("train", 0.7), val_tag: split_ratios.get("val", 0.2), test_tag: split_ratios.get("test", 0.1)})
            dataset.save()
        else:
            print("Dataset already contains split tags. Skipping new split.")
            
    return dataset

# =============================================================================
# AUGMENTATION LOGIC
# =============================================================================
def augment_samples(dataset, filter_tags=None, new_dataset_name=None, output_dir=None, num_aug=1):
    """
    Apply augmentations to samples matching `filter_tags`.
    If `filter_tags` is None, augment ALL samples.
    If `new_dataset_name` is provided:
       - If exists: Append new samples to it.
       - If not exists: Create it and add new samples.
    If `new_dataset_name` NOT provided:
       - Add new samples back to `dataset`.
    
    Augmented sample behavior:
       - Created on disk at `data/augmented/{dest_dataset_name}/...`
       - ID: New unique ID.
       - Tags: Copied from original sample + "augmented".
    """
    # 1. Select Samples
    if filter_tags:
        print(f"Augmenting samples with tags: {filter_tags}")
        # FiftyOne's match_tags with a list means "samples that have ANY of these tags".
        # If we want "samples that have ALL of these tags", we chain them.
        view = dataset
        for tag in filter_tags:
            view = view.match_tags(tag)
    else:
        print("Augmenting ALL samples in dataset.")
        view = dataset
    
    if len(view) == 0:
        print("No samples found to augment.")
        return dataset

    # 2. Determine Destination Dataset
    dest_dataset = dataset
    if new_dataset_name:
        if new_dataset_name == dataset.name:
             dest_dataset = dataset
        elif new_dataset_name in fo.list_datasets():
            print(f"Destination dataset {new_dataset_name} exists. Loading to append...")
            dest_dataset = fo.load_dataset(new_dataset_name)
        else:
            print(f"Creating new dataset {new_dataset_name} for augmented samples...")
            dest_dataset = fo.Dataset(name=new_dataset_name)
            dest_dataset.persistent = True # Make sure it stays

    # Define storage for augmented images
    if output_dir:
        aug_dir = output_dir
    else:
        aug_dir = os.path.join("data", "augmented", dest_dataset.name)
    os.makedirs(aug_dir, exist_ok=True)
    
    transform = get_augmentation_pipeline()
    
    new_samples = []
    
    for sample in tqdm(view, desc="Augmenting"):
        img_path = sample.filepath
        image = cv2.imread(img_path)
        if image is None: 
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Extract Bboxes
        # FiftyOne: [x, y, w, h] normalized
        # Albumentations YOLO: [xc, yc, w, h] normalized
        bboxes = []
        labels = []
        
        if sample.ground_truth:
            for det in sample.ground_truth.detections:
                x, y, wa, ha = det.bounding_box
                xc = x + wa / 2
                yc = y + ha / 2
                bboxes.append([xc, yc, wa, ha])
                labels.append(det.label)
        
        for i in range(num_aug):
            try:
                aug = transform(image=image, bboxes=bboxes, class_labels=labels)
                aug_img = aug['image']
                aug_bboxes = aug['bboxes']
                
                # Save Image
                filename = f"aug_{i}_{sample.id}_{os.path.basename(img_path)}"
                save_path = os.path.join(aug_dir, filename)
                cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                
                # Create Detections
                # Albu YOLO -> FiftyOne [x, y, w, h] normalized
                new_dets = []
                for bbox, label in zip(aug_bboxes, aug['class_labels']):
                    xc, yc, wa, ha = bbox
                    x = xc - wa / 2
                    y = yc - ha / 2
                    new_dets.append(fo.Detection(label=label, bounding_box=[x, y, wa, ha]))
                
                # Create Sample
                s = fo.Sample(filepath=os.path.abspath(save_path))
                s["ground_truth"] = fo.Detections(detections=new_dets)
                
                # Copy tags and add 'augmented'
                new_tags = list(sample.tags) if sample.tags else []
                if "augmented" not in new_tags:
                    new_tags.append("augmented")
                s.tags = new_tags
                
                new_samples.append(s)
                
            except Exception as e:
                print(f"Augmentation failed for {img_path}: {e}")
                
    if new_samples:
        dest_dataset.add_samples(new_samples)
        dest_dataset.save()
        print(f"Added {len(new_samples)} augmented samples to {dest_dataset.name}.")
    else:
        print("No samples generated.")
        
    return dest_dataset

# =============================================================================
# EXPORT LOGIC
# =============================================================================
def export_pipeline(
    dataset,
    export_dir,
    classes=None,
    train_tag="train",
    val_tag="val",
    test_tag="test",
    export_tags=None,
    copy_images=False,
    include_confidence=False,
):
    """
    Export dataset to hybrid YOLO + COCO format.

    Export Modes:
        - Default (copy_images=False): Creates symlinks to original images in
          ``images/{split}/`` and generates fresh 5-column YOLO labels in
          ``labels/{split}/``. FiftyOne auto-generates ``dataset.yaml``.
          YOLO's ``img2label_paths()`` naturally resolves symlinked image paths
          to the exported labels.
        - Portable (copy_images=True): Copies images into ``images/{split}/``.
          Produces a fully self-contained dataset for sharing/upload.

    Args:
        dataset: FiftyOne dataset.
        export_dir: Destination directory for the export.
        classes: Optional list of class names. If None, inferred from data.
        train_tag: Tag identifying training samples.
        val_tag: Tag identifying validation samples.
        test_tag: Tag identifying test samples.
        export_tags: Optional list of tags. Only samples with ALL tags are
            exported.
        copy_images: If True, copies images (self-contained). If False,
            creates symlinks (zero disk bloat for images).
        include_confidence: If True, writes 6-column YOLO labels (with
            confidence score). Default False writes standard 5-column.
    """
    export_dir = os.path.abspath(export_dir)
    export_mode = True if copy_images else "symlink"
    mode_label = "copy" if copy_images else "symlink"
    print(f"Exporting to {export_dir} (media mode: {mode_label})...")

    split_map = {"train": train_tag, "val": val_tag, "test": test_tag}

    if export_tags:
        print(f"Filtering export for samples containing ALL tags: {export_tags}")
        base_view = dataset
        for t in export_tags:
            base_view = base_view.match_tags(t)
    else:
        base_view = dataset

    for std_split, user_tag in split_map.items():
        split_view = base_view.match_tags(user_tag)
        if len(split_view) == 0:
            continue

        print(f"  [{std_split}] {len(split_view)} samples")

        # --- YOLO export (labels + images or symlinks) -----------------------
        # FiftyOne regenerates labels from in-memory Detection objects.
        # With export_media="symlink", it creates symlinks in images/{split}/
        # and writes clean labels in labels/{split}/, then auto-writes
        # dataset.yaml with correct relative paths.
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=classes,
            split=std_split,
            export_media=export_mode,
            include_confidence=include_confidence,
        )

        # --- COCO JSON export (for RF-DETR) ----------------------------------
        json_path = os.path.join(export_dir, f"_annotations_{std_split}.coco.json")
        split_view.export(
            labels_path=json_path,
            dataset_type=fo.types.COCODetectionDataset,
            label_field="ground_truth",
            classes=classes,
            export_media=False,
        )

        # Normalise file_name to basenames in the COCO JSON.
        # Guard against duplicate basenames — if found, keep full paths and warn.
        if os.path.exists(json_path):
            _fix_coco_filenames(json_path)

    print("Export complete.")


def _fix_coco_filenames(json_path):
    """Normalise ``file_name`` in a COCO JSON to basenames.

    If duplicate basenames are detected, full paths are preserved and a
    warning is printed.  This prevents silent dataset corruption when
    images from different directories share the same filename.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    basenames = [os.path.basename(img["file_name"]) for img in data["images"]]
    counts = Counter(basenames)
    duplicates = {name: cnt for name, cnt in counts.items() if cnt > 1}

    if duplicates:
        print(
            f"  WARNING: Duplicate basenames detected in {os.path.basename(json_path)}: "
            f"{duplicates}. Keeping full paths to avoid corruption."
        )
    else:
        for img in data["images"]:
            img["file_name"] = os.path.basename(img["file_name"])

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset Manager: Load, Augment, Export.")
    
    # Init / Load
    parser.add_argument("--dataset-dir", default="data/raw", help="Path to raw YOLO dataset")
    parser.add_argument("--name", default="my_dataset", help="FiftyOne dataset name")
    parser.add_argument("--split", action="store_true", help="Perform random split if not present")
    
    # Augmentation
    parser.add_argument("--augment", action="store_true", help="Run augmentation")
    parser.add_argument("--augment-tags", nargs='+', help="List of tags to augment (e.g. 'train'). If empty, augments ALL.")
    parser.add_argument("--output-dataset", type=str, help="Name of destination dataset for augmented samples. Default: Same.")
    parser.add_argument("--output-dir", type=str, help="Directory to save augmented images. Default: data/augmented/<dataset_name>")
    
    # Export
    parser.add_argument("--export-dir", type=str, help="Directory to export final dataset")
    parser.add_argument("--export-tags", nargs='+', help="Filter export to samples having ALL these tags (e.g. 'augmented')")
    parser.add_argument("--copy-images", action="store_true", help="Copy images to export dir (self-contained). Default: symlink to originals (zero disk bloat)")
    parser.add_argument("--include-confidence", action="store_true", help="Export 6-column YOLO labels (with confidence score). Default: 5-column")
    
    # View
    parser.add_argument("--view", action="store_true", help="Launch FiftyOne App after processing")
    
    # Custom Tag Mappings
    parser.add_argument("--train-tag", default="train", help="Tag identifying training samples (default: train)")
    parser.add_argument("--val-tag", default="val", help="Tag identifying validation samples (default: val)")
    parser.add_argument("--test-tag", default="test", help="Tag identifying test samples (default: test)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    ratios = {"train": 0.7, "val": 0.2, "test": 0.1} if args.split else None
    dataset = load_or_create_dataset(
        args.dataset_dir,
        args.name,
        split_ratios=ratios,
        train_tag=args.train_tag,
        val_tag=args.val_tag,
        test_tag=args.test_tag,
    )

    if args.augment:
        dataset = augment_samples(
            dataset,
            filter_tags=args.augment_tags,
            new_dataset_name=args.output_dataset,
            output_dir=args.output_dir,
            num_aug=1,
        )

    if args.export_dir:
        classes = dataset.default_classes
        if not classes:
            classes = dataset.distinct("ground_truth.detections.label")
        export_pipeline(
            dataset,
            args.export_dir,
            classes=classes,
            train_tag=args.train_tag,
            val_tag=args.val_tag,
            test_tag=args.test_tag,
            export_tags=args.export_tags,
            copy_images=args.copy_images,
            include_confidence=args.include_confidence,
        )

    if args.view:
        print(f"Launching FiftyOne App for dataset: {dataset.name}")
        session = fo.launch_app(dataset)
        session.wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
