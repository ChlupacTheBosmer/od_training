import argparse
import os
import cv2
import supervision as sv
from pathlib import Path
import numpy as np
from typing import Union, List


from ..utility.runtime_config import ensure_local_config

ensure_local_config()

# Import models
from ultralytics import YOLO

try:
    from rfdetr import (
        RFDETRNano,
        RFDETRSmall,
        RFDETRMedium,
        RFDETRLarge,
        RFDETRXLarge,
        RFDETR2XLarge,
        RFDETRBase
    )
    RFDETR_AVAILABLE = True
    RFDETR_MAP = {
        "rfdetr_nano": RFDETRNano,
        "rfdetr_small": RFDETRSmall,
        "rfdetr_medium": RFDETRMedium,
        "rfdetr_large": RFDETRLarge,
        "rfdetr_xlarge": RFDETRXLarge,
        "rfdetr_2xlarge": RFDETR2XLarge,
        "rf-detr-resnet50": RFDETRMedium,
        "rf-detr-base": RFDETRBase,
    }
except ImportError:
    RFDETR_AVAILABLE = False
    RFDETR_MAP = {}
    print("Warning: RF-DETR not found or imports failed. RF-DETR inference will not be available.")

def load_rfdetr_model(model_name: str, rfdetr_arch: str = None):
    """
    Load RF-DETR model.
    If model_name is a key in map, instantiate it (downloads weights).
    If model_name is a path, instantiate arch (default Medium) with pretrain_weights=path.
    """
    if not RFDETR_AVAILABLE:
        raise ImportError("RF-DETR package not available.")

    # 1. Check if model_name is a known architecture key
    if model_name.lower() in RFDETR_MAP:
        print(f"Loading standard RF-DETR model: {model_name}")
        return RFDETR_MAP[model_name.lower()]()

    # 2. Assume it's a path
    path = Path(model_name)
    if not path.exists():
        # Fallback check if it maps to a class name roughly
        for k, v in RFDETR_MAP.items():
            if k in model_name.lower():
                 print(f"Loading standard RF-DETR model (inferred): {k}")
                 return v()
        raise FileNotFoundError(f"Model path not found: {model_name}")
    
    # It is a path to custom weights
    print(f"Loading custom RF-DETR weights from: {path}")
    
    # Determine architecture
    if rfdetr_arch and rfdetr_arch.lower() in RFDETR_MAP:
        arch_cls = RFDETR_MAP[rfdetr_arch.lower()]
    elif "nano" in model_name.lower():
         arch_cls = RFDETRNano
    elif "small" in model_name.lower():
         arch_cls = RFDETRSmall
    elif "large" in model_name.lower():
         arch_cls = RFDETRLarge
    else:
        print("Warning: Architecture not specified and could not infer from filename. Defaulting to RFDETRMedium.")
        arch_cls = RFDETRMedium
        
    return arch_cls(pretrain_weights=str(path))


def run_inference(
    source: str,
    model_name: str,
    model_type: str,
    rfdetr_arch: str = None,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    show: bool = False,
    save_dir: str = None,
):
    # Load Model
    if model_type.lower() == 'yolo':
        print(f"Loading YOLO model: {model_name}")
        model = YOLO(model_name)
    elif model_type.lower() == 'rfdetr':
        model = load_rfdetr_model(model_name, rfdetr_arch)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Handle Source
    source_path = Path(source)

    frame_generator = None
    is_video = False
    source_fps = 30  # default, overridden for video sources
    image_filenames = []  # track original filenames for directory mode

    if source == '0' or source.startswith('rtsp'):
        is_video = True
        frame_generator = sv.get_video_frames_generator(source)
    elif source_path.is_file():
        ext = source_path.suffix.lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            is_video = True
            frame_generator = sv.get_video_frames_generator(source)
            # Read FPS from source
            cap = cv2.VideoCapture(source)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                source_fps = fps
            cap.release()
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            frame = cv2.imread(source)
            if frame is None:
                raise ValueError(f"Could not read image: {source}")
            frame_generator = iter([frame])
            image_filenames = [source_path.stem]
    elif source_path.is_dir():
         exclude_files = ['.DS_Store']
         image_files = sorted(
             p for p in source_path.glob('*')
             if p.suffix.lower() in ['.jpg', '.jpeg', '.png'] and p.name not in exclude_files
         )
         if not image_files:
             print(f"No images found in {source_path}")
             return

         image_filenames = [p.stem for p in image_files]

         def dir_gen():
             for p in image_files:
                 f = cv2.imread(str(p))
                 if f is not None:
                     yield f
         frame_generator = dir_gen()

    if frame_generator is None:
        raise ValueError(f"Could not process source: {source}")

    # Output setup
    video_writer = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    if show:
        cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)

    print(f"Starting inference on {source}...")
    
    for i, frame in enumerate(frame_generator):
        # Inference
        if model_type.lower() == 'yolo':
            results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
        elif model_type.lower() == 'rfdetr':
            # RF-DETR returns sv.Detections naturally
            detections = model.predict(frame, threshold=conf_threshold)
            if isinstance(detections, list):
                detections = detections[0]
            # No manual NMS usually needed for DETR
            
        # Annotate
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Show
        if show:
            cv2.imshow("Inference", annotated_frame)
            if cv2.waitKey(1) == ord('q'):
                break
        
        # Save
        if save_dir:
            if is_video:
                 if video_writer is None:
                     h, w = frame.shape[:2]
                     out_path = os.path.join(save_dir, "output_video.mp4")
                     video_writer = cv2.VideoWriter(
                         out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                         source_fps, (w, h),
                     )
                 video_writer.write(annotated_frame)
            else:
                # Use original filename with _annotated suffix
                if i < len(image_filenames):
                    fname = f"{image_filenames[i]}_annotated.jpg"
                else:
                    fname = f"frame_{i:04d}.jpg"
                out_path = os.path.join(save_dir, fname)
                cv2.imwrite(out_path, annotated_frame)

    if video_writer:
        video_writer.release()
    if show:
        cv2.destroyAllWindows()
    
    print("Inference complete.")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference using YOLO or RF-DETR models.")
    parser.add_argument("--source", type=str, required=True, help="Path to image, video, directory, or '0' for webcam")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., yolo11n.pt, rfdetr_nano) or path to weights")
    parser.add_argument("--type", type=str, required=True, choices=['yolo', 'rfdetr'], help="Model type: 'yolo' or 'rfdetr'")
    parser.add_argument("--rfdetr-arch", type=str, help="RF-DETR architecture if loading custom weights (e.g. rfdetr_medium)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (for NMS)")
    parser.add_argument("--show", action="store_true", help="Display results in window")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save results (default: no saving)")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_inference(
        source=args.source,
        model_name=args.model,
        model_type=args.type,
        rfdetr_arch=args.rfdetr_arch,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        show=args.show,
        save_dir=args.save_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
