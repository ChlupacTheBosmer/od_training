# Research Notes

## YOLO (Ultralytics)

### Versions
*   **YOLOv11**: Released late 2024. Transformer-based backbone, dynamic head, NMS-free training options.
*   **YOLO26**: Released Jan 2026. Latest generation. Optimized for edge/low-power but scalable.
    *   **Features**: Fully end-to-end, NMS-free, enhanced small-object recognition, faster CPU inference.
    *   **Architecture**: Transformer backbone, dynamic head.

### Requirements
*   **Package**: `ultralytics` (latest version).
*   **Python**: 3.8+
*   **Hardware**: CUDA-compatible GPU (4GB+ VRAM recommended).
*   **Data Format**: Standard YOLO format (images in dir, labels in dir with `.txt` files containing `class_id x_center y_center width height`).

## RF-DETR (Roboflow)

### Overview
*   **Source**: [Roboflow RF-DETR](https://github.com/roboflow/rf-detr)
*   **Nature**: Transformer-based object detector (DETR family).
*   **Usage**: Can be used standalone (`pip install rfdetr`) or with Roboflow platform.

### Requirements
*   **Package**: `rfdetr`, `roboflow`.
*   **Data Format**: **Microsoft COCO JSON**.
    *   Structure: `train/`, `valid/`, `test/` directories, each with `_annotations.coco.json`.
    *   **Note**: Incompatible directly with YOLO format. Conversion required.

### Standalone Training
*   Can fine-tune from pre-trained COCO checkpoints (`RFDETRBase`).
*   Training script uses `rfdetr.model.train()` or similar API.

## Data Conversion
*   **Need**: YOLO (txt) <-> COCO (json).
*   **Solution**:
    *   Use `roboflow` pip package for easy download/conversion from platform.
    *   Implement local conversion utilities for offline datasets to avoid uploading everything to Roboflow if privacy/bandwidth is a concern.

## Environment & Kubernetes
*   **Base**: User provided image.
*   **Additions**: `requirements.txt` installing `ultralytics`, `rfdetr`, `roboflow`, `clearml`.
*   **Pathing**: Must use relative paths or environment variables for data directories to ensure portability between local Mac and K8s Pod.

## YOLOv11 & YOLO26 Technical Specifics (Latest Research)

### Model Loading & Naming
*   **Initialization**: Use standard Ultralytics naming.
    *   **YOLOv11**: `YOLO("yolo11n.pt")`, `YOLO("yolo11s.pt")`, etc.
    *   **YOLO26**: `YOLO("yolo26n.pt")`, `YOLO("yolo26s.pt")`, `YOLO("yolo26m.pt")`, `YOLO("yolo26l.pt")`, `YOLO("yolo26x.pt")`.
    *   **Task Variants**: Available as `yolo26n-seg.pt`, `yolo26n-pose.pt`, etc.

### YOLO26 Specifics
*   **Architecture**:
    *   **Dual-Head**: 
        *   **One-to-One Head (Default)**: Optimized for NMS-free inference (end-to-end).
        *   **One-to-Many Head**: Used for denser supervision during training.
    *   **Efficiency**: Removed Distribution Focal Loss (DFL) for faster edge inference.
*   **Training & Hyperparameters**:
    *   **Optimizer**: Defaults to **MuSGD** (Muon + SGD hybrid) for stability.
    *   **Losses**: Introduces **ProgLoss** (Progressive Loss Balancing) and **STAL** (Small Tiny Attention Loss).
    *   **NMS-Free**: Enabled by default.
*   **Flags**:
    *   `end2end=True`: Use during prediction/export to enforce the NMS-free one-to-one head.

### Hardware Optimization (NVIDIA GPU / Kubernetes)
*   **Recommendation**:
    *   **Format**: **TensorRT 10** (`.engine`) is the benchmark standard for NVIDIA T4/L4 GPUs.
    *   **Export Command**: 
        ```python
        model.export(format="engine", end2end=True) 
        ```
    *   **ONNX**: Supported as a fallback.
*   **Performance**: The simplified head architecture reduces GPU overhead.

### Metrics & Logging
*   **New Metric**: `mAP val 50-95(e2e)` - Measures performance of the end-to-end head without NMS.
*   **Logging**: Tracks both one-to-one and one-to-many head contributions during training.

## RF-DETR Standalone Workflow (Research)

### 1. Installation
The `rfdetr` package is the core library. It requires strict Python version adherence due to dependency constraints (mainly PyTorch and associated compile tools).

*   **Command**:
    ```bash
    pip install rfdetr supervision roboflow
    ```
    *   **Note**: `supervision` is highly recommended for handling predictions and metrics. `roboflow` is needed if you want to download datasets from the platform, but `rfdetr` training can run on local files.
*   **Python Version**: **3.10** (Recommended). Support for 3.11 is experimental; 3.12 is likely to fail.
*   **Hardware**: Standard CUDA-enabled GPU (NVIDIA).

### 2. Training Script
RF-DETR uses a high-level API similar to other modern detectors. It expects data in **COCO JSON** format.

**Dataset Structure (Local)**:
```
/path/to/dataset/
├── train/
│   ├── _annotations.coco.json
│   └── image1.jpg ...
├── valid/
│   ├── _annotations.coco.json
│   └── imageX.jpg ...
└── test/
    ├── _annotations.coco.json
    └── imageY.jpg ...
```

**Complete Training Script (`train_rfdetr_standalone.py`)**:
```python
import os
from rfdetr import RFDETR
from roboflow import Roboflow
import argparse

def train_standalone(dataset_dir, model_type="rf-detr-resnet50", epochs=50, batch_size=4, lr=1e-4):
    """
    Trains RF-DETR on a local COCO dataset.
    """
    print(f"Loading model: {model_type}")
    # Initialize the model. This typically downloads pretrained COCO weights.
    model = RFDETR(model_type)

    print(f"Starting training on {dataset_dir}...")
    # The .train() method handles the loop. 
    # args may vary slightly by version, checking `help(model.train)` is recommended.
    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )
    
    # Save is usually automatic, but explicit save can be done
    # save_path = os.path.join("runs", "weights")
    # model.save(save_path)
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory (must contain train/valid/test subfolders with COCO json)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", type=str, default="rf-detr-resnet50", help="Model architecture (e.g. rf-detr-resnet50, rf-detr-resnet101)")
    args = parser.parse_args()

    train_standalone(args.dataset, args.model, args.epochs, args.batch_size)
```

### 3. Evaluation
Evaluation is often manual or semi-automated using the `supervision` library if the `model.val()` method is not sufficient or available.

**Testing Workflow**:
1.  Load the trained model (typically `model = RFDETR("path/to/best_weights.pt")`).
2.  Iterate through the `test` set images.
3.  Predict and compare with ground truth using `supervision.DetectionDataset` and metrics.

### 4. Integration with ClearML
RF-DETR does not appear to have a built-in "ClearMLCallback". You must wrap the training script with ClearML's automatic logging or manual reporting.

**Integration Strategy**:
```python
from clearml import Task

# Initialize BEFORE imports if possible, or right at start
task = Task.init(project_name="Object Detection/RF-DETR", task_name="rf-detr-training-run")

# Connect arguments
params = {
    "epochs": 50,
    "batch_size": 8,
    "model": "rf-detr-resnet50"
}
task.connect(params)

# ... Run Training ...
model.train(...)
```

### 5. Artifacts
*   **Format**: `.pt` (PyTorch state dictionary).
*   **Location**: By default, `rfdetr` creates an output directory (e.g., `./runs/{date}_{time}`).

### 6. Roboflow Platform Interaction (Optional)
If you wish to use the platform for dataset management or hosting weights:

*   **Download Dataset**:
    ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("workspace-name").project("project-name")
    dataset = project.version(1).download("coco") # Downloads to folder, returns path
    print(dataset.location)
    ```

*   **Upload Weights**:
    ```python
    # Upload trained weights for hosted inference api
    project.version(1).deploy(model_type="rf-detr", model_path="runs/weights/best.pt")
    ```

### 7. Logging, Checkpoints & Validation (Advanced)

#### Native Logging
RF-DETR supports **TensorBoard** and **Weights & Biases (W&B)** out-of-the-box.
*   **Installation**: `pip install "rfdetr[metrics]"`
*   **Usage**:
    ```python
    model.train(
        ...,
        tensorboard=True,
        wandb=True,
        # optional: project="my-project", name="run-name" for W&B
    )
    ```
*   **CSV/Text**: Basic logs are printed to stdout, which can be piped to a file. Native CSV logging is not explicitly documented as a standalone argument but may be generated by the underlying trainer.

#### Custom Integration (No Native Callbacks)
The high-level `model.train()` does not expose a standard list of callbacks (like Keras or Lightning) for easy injection.
*   **Strategy**: To use **ClearML** or other custom loggers, you must wrap the training script (as shown in section 4) or start the generic task logger at the beginning of your script.
*   **Advanced**: The internal library may support `model.callback_on_fit_epoch()` but this is experimental. Wrapping is the stable "black-box" solution.

#### Checkpointing
RF-DETR saves specific checkpoint files in the `output_dir` (default: `runs/{date}_{time}`):
*   **`checkpoint_best_ema.pth`**: Best model based on validation mAP using **Exponential Moving Average** weights (smoother, usually better).
*   **`checkpoint_best_regular.pth`**: Best model using raw weights.
*   **`checkpoint_best_total.pth`**: The final **recommended** weights for inference (automatically selected between EMA and Regular based on performance). **Use this as your `best.pt`.**

#### Validation Loop
*   **In-Training**: Validation runs automatically at the end of epochs. It calculates mAP@50 and mAP@50:95.
*   **Standalone (`model.val()`)**: **Does NOT exist** as a public API method in the same way as YOLO.
*   **Programmatic Evaluation**: To get mAP scores on a test set programmatically:
    1.  Use **Supervision** (`pip install supervision`).
    2.  Run inference on your test set.
    3.  Calculate metrics using `supervision.DetectionDataset` and `MeanAveragePrecision`.
    
    ```python
    import supervision as sv
    from rfdetr import RFDETR
    
    # Load model
    model = RFDETR("runs/path/to/checkpoint_best_total.pth")
    
    # Load dataset (COCO format)
    ds = sv.DetectionDataset.from_coco(
        images_directory_path="test/images",
        annotations_path="test/_annotations.coco.json"
    )
    
    # Run Inference Callback
    def callback(image):
        result = model.predict(image)
        return sv.Detections.from_rfdetr(result)
        
    # Benchmark
    mAP = sv.MeanAveragePrecision.benchmark(
        dataset=ds,
        callback=callback
    )
    print(f"mAP50: {mAP.map50}, mAP50-95: {mAP.map50_95}")
    ```

## Data Interchange & Verification

### 1. Conversion Tools

Robust conversion between YOLO and COCO formats is critical for interoperability between different models (e.g., YOLOv8 and RF-DETR).

#### YOLO -> COCO
*   **Recommended Library**: `globox` or `fiftyone`.
*   **Globox**: Lightweight, supports YOLOv5/v8, efficient.
    ```bash
    pip install globox
    ```
    ```python
    from globox import AnnotationSet

    # Read YOLO annotations (requires images to get dimensions)
    annotations = AnnotationSet.from_yolo_v5(
        folder="path/to/labels",
        image_folder="path/to/images"
    )
    annotations.save_coco("annotations.json")
    ```
*   **FiftyOne**: Heavier but extremely robust for complex datasets and visualization.

#### COCO -> YOLO
*   **Recommended Library**: `ultralytics` (Native support).
*   **Usage**:
    ```python
    from ultralytics.data.converter import convert_coco

    convert_coco(
        labels_dir='path/to/coco/annotations/',
        use_segments=False,
        use_keypoints=False,
        cls91to80=False
    )
    ```

### 2. Dataset Validation

Before training, datasets must be validated to prevent silent failures or poor performance.

#### Key Checks & Efficiency
*   **Corrupt Images**:
    *   **Method**: Try opening with `PIL` or `OpenCV`.
    *   **Script Snippet**:
        ```python
        from PIL import Image
        from pathlib import Path

        def check_images(img_dir):
            for img_path in Path(img_dir).glob('*.*'):
                try:
                    with Image.open(img_path) as img:
                        img.verify() # fast check
                except Exception as e:
                    print(f"Corrupt image: {img_path} - {e}")
        ```
*   **Missing Labels**:
    *   Check if every image has a corresponding label file (for YOLO) or entry (for COCO).
    *   Check for empty label files if the dataset shouldn't contain background images.
*   **Negative/Invalid Bounding Boxes**:
    *   **YOLO**: Check if `0 <= x_center, y_center, width, height <= 1`.
    *   **COCO**: Check if `x, y >= 0` and `x + w <= image_width`, `y + h <= image_height`.

### 3. Roboflow SDK Download Snippet

Downloading a dataset in both formats to serve as ground truth for testing converters.

```python
from roboflow import Roboflow

# Initialize
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("project-name")
version = project.version(1) # Replace with specific version

# Download YOLOv8 format
dataset_yolo = version.download("yolov8")

# Download COCO format (creates distinct folder)
dataset_coco = version.download("coco")

print(f"YOLO path: {dataset_yolo.location}")
print(f"COCO path: {dataset_coco.location}")
```

## YOLO26 Fine-tuning Strategy (Research)

### 1. Freezing Layers (Transformer Backbone)
*   **Mechanism**: The `freeze` argument remains valid but operates on *Transformer Blocks*.
*   **Usage**: 
    *   `freeze=10`: Freezes the first 10 blocks.
    *   `freeze='backbone'`: **New in YOLO26**. Automatically identifies and freezes the entire transformer backbone. Recommended for small custom datasets to prevent overfitting.

### 2. Hyperparameters (MuSGD Optimizer)
*   **Optimizer**: Default `MuSGD` (Muon + SGD hybrid) for stability.
*   **Learning Rate**:
    *   **Standard**: `lr0=0.01` is often too high for fine-tuning.
    *   **Recommendation**: Start with **`lr0=0.002`** or **`lr0=0.0025`**.
*   **Momentum**: Keep high (`0.98` or `0.99`) for stability.

### 3. Data Augmentation (End-to-End Sensitivity)
*   **Challenge**: The **End-to-End (One-to-One)** head uses strict bipartite matching, which is sensitive to "noisy" augmentations like heavy Mosaic.
*   **Strategy**:
    *   **Mosaic**: **Reduce** intensity or disable earlier. Configure **`close_mosaic=20`** (last 20 epochs) to let the model settle on realistic distributions.
    *   **MixUp**: generally **disable** (`mixup=0.0`) for the One-to-One head unless the dataset is very large (>50k images).

## RF-DETR & Ultralytics Compatibility (Dependency Research)

### 1. Version Constraints

| Package | `rfdetr` | `ultralytics` | **Resolution / Pin** |
| :--- | :--- | :--- | :--- |
| **Python** | `>=3.10` | `>=3.8` | **3.10** |
| **PyTorch** | `>=1.13, <=2.8` | `>=1.8` | **Latest Stable (~2.4)** |
| **Numpy** | Issue with 2.0 | `>=1.23.5` | **`<2.0.0`** (Critical) |
| **Pydantic** | Flexible | V2 support fragile | **`<2.0.0`** (Safest) |

### 2. Installation Strategy
1.  **Base Image**: Start with an official PyTorch image (e.g., `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`). This ensures Python 3.10+ and correctly compiled Torch/CUDA.
2.  **Pre-install Pins**:
    ```bash
    pip install "numpy<2.0.0" "pydantic<2.0.0"
    ```
3.  **Install Packages**:
    ```bash
    pip install ultralytics rfdetr
    ```


## Roboflow Local Capabilities Research

### Objective
Determine if the `roboflow` Python package supports local dataset management, processing, augmentations, and versioning without uploading data to the Roboflow platform.

### Executive Summary
**No, the `roboflow` Python package does not support local-only dataset management, augmentation, or versioning.**

The `roboflow` package functions primarily as a **client SDK** (Software Development Kit) that interacts with the Roboflow API. It does not contain the core logic for image processing, augmentation, or dataset version control; these operations are performed on the Roboflow servers (Cloud) or on a self-hosted Roboflow instance (On-Premise).

### Findings

#### 1. Architecture: API Client, Not Standalone Tool
*   **Code Analysis**: The codebase (`roboflow-python`) consists almost entirely of API wrappers.
    *   `Project.generate_version(settings=...)`: Sends a JSON payload describing desired augmentations to the API. It does **not** process images locally.
    *   `Project.version()`: Retrieves metadata about a version from the API.
    *   `Project.upload()`: Sends binary image data to the server.
*   **"Local" Features**: References to "local" in the codebase (e.g., `local` parameter in `Version`, `uselocal` script) refer to connecting the SDK to a **locally hosted instance of the Roboflow API** (e.g., Docker container), not to offline Python function calls.

#### 2. Local Dataset Management & Versioning
*   **Management**: There is no local database or file-system-based manager included in the package. You cannot "create a project" or "add an image" to a local-only registry using `roboflow` classes.
*   **Versioning**: Versioning is handled by the Roboflow backend. You cannot "commit" a dataset version locally.

#### 3. Processing & Augmentation
*   **Implementation**: A grep search for "augmentation" in the source code reveals it is only used to construct API payloads. There are no image processing libraries (like `opencv` or `PIL` based manipulation functions for augmentation) exposed or implemented for user data.
*   **Recommendation**: To perform these steps locally, you must use separate libraries.

### Alternatives for Local Workflow
Since `roboflow` cannot be used for this purpose in an offline/local-only manner, the following alternatives are recommended:

*   **Data Management**: Use a directory structure (YOLO format or COCO format) manually or via `fiftyone`.
*   **Augmentation**: Use **`albumentations`** (industry standard for detection) or `torchvision`.
*   **Versioning**: Use Git (via DVC - Data Version Control) or simply folder naming conventions (e.g., `dataset_v1`, `dataset_v2`).
*   **Visualization/Logic**: Use **`supervision`** (also by Roboflow) which **is** a local library for processing predictions and visualizing data, though it doesn't manage dataset *versions*.

## Supervision (Roboflow) Research

### Overview
[Supervision](https://github.com/roboflow/supervision) is an open-source Python library that serves as a versatile "Swiss Army knife" for computer vision tasks. It is designed to be model-agnostic and acts as the glue between model inference and application logic.

### Core Capabilities
*   **Annotators**: Extensive library of highly customizable annotators (Box, Mask, Label, Blur, Pixelate, Trace, Heatmap) to visualize detections and tracking on images or video.
*   **Detections Handling**: Unified `Detections` class to handle bounding boxes, masks, confidence scores, and class IDs from various libraries (Ultralytics, Transformers, MMDetection, etc.).
*   **Zones & Counting**: Built-in logic for defining polygon zones and counting objects that enter/exit or stay within them (e.g., `LineZone`, `PolygonZone`).
*   **Object Tracking**: Wrappers for tracking algorithms (like ByteTrack) to assign generic IDs to detections across frames.
*   **Filtering & Post-Processing**: Utilities to filter detections by area, confidence, or class, and perform Non-Max Suppression (NMS) or tracking smoothing.

### Integration with RF-DETR (Native Support Confirmed)
**Crucial Finding**: The `rfdetr` library (specifically `rfdetr.RFDETR.predict`) **natively returns** `supervision.Detections` objects.
*   **Source**: `rfdetr/detr.py`.
*   **Implication**: No custom conversion adapter is needed. You can pass the output of `model.predict()` directly to Supervision annotators or metrics tools.
*   **Example**:
    ```python
    from rfdetr import RFDETR
    import supervision as sv

    model = RFDETR("rf-detr-resnet50")
    detections = model.predict("image.jpg") # Returns sv.Detections
    
    annotator = sv.BoxAnnotator()
    annotated_frame = annotator.annotate(scene=image, detections=detections)
    ```

## Albumentations Research

### Overview
[Albumentations](https://albumentations.ai/) is the industry-standard Python library for image augmentation. It is favored for its speed and flexibility.

### Integration with FiftyOne
**FiftyOne has a plugin for Albumentations** which must be installed separately from the pip package.
*   **Installation**:
    ```bash
    fiftyone plugins download https://github.com/voxel51/fiftyone-plugins --plugin-names albumentations
    ```
*   **Workflow**:
    1.  **Define Pipeline**: Create an Albumentations composition in Python.
    2.  **Visualize**: Use the FiftyOne App to apply this pipeline to your dataset samples dynamically *without* modifying the source files on disk initially.
    3.  **Generate**: Once satisfied, use the pipeline to generate a persistent training dataset (e.g., exporting to YOLO format with augmentations applied).

### Example: Albumentations Pipeline for Detection
```python
import albumentations as A

transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

# Research: FiftyOne & Albumentations Workflow for Object Detection

**Date:** 2026-02-10
**Status:** In Progress
**Objective:** Establish a robust pipeline for curating datasets in FiftyOne, applying Albumentations (crop, resize, etc.), and exporting for RF-DETR (COCO format) and YOLO training.

---

## 1. Executive Summary
The optimal workflow involves leveraging the [FiftyOne Albumentations Plugin](https://github.com/jacobmarks/fiftyone-albumentations-plugin) to apply and **persist** augmentations directly within the FiftyOne database before export. This ensures:
1.  **Visual Verification**: You can preview augmentations in the App before committing them.
2.  **Metadata Management**: FiftyOne handles the complex logic of transforming bounding boxes (Detections) to match geometric augmentations (crops, resizes).
3.  **Unified Export**: Once augmented samples are saved in FiftyOne, you can export to multiple formats (YOLO, RF-DETR/COCO) from the same source of truth, ensuring dataset consistency.

## 2. Prerequisites
- **FiftyOne**: `pip install fiftyone`
- **Albumentations**: `pip install albumentations`
- **FiftyOne Albumentations Plugin**:
  ```bash
  fiftyone plugins download https://github.com/jacobmarks/fiftyone-albumentations-plugin
  ```

---

## 3. Step-by-Step Workflow

### Phase 1: Curate & Split (in FiftyOne)
Before augmenting, define your base dataset structure (Train/Val/Test) and filter unwanted samples.

**A. Filtering by Tags**
Use FiftyOne `Views` to select samples.
```python
import fiftyone as fo

dataset = fo.load_dataset("my_dataset")

# Filter only samples with specific tags (e.g., 'confirmed_ground_truth')
view = dataset.match_tags("confirmed_ground_truth")

# Filter out unrelated classes if necessary
# output_field='ground_truth' assumes your labels are in this field
view = view.filter_labels("ground_truth", fo.Filter("label").in_list(["person", "car"]))
```

**B. Creating Splits (Train/Val/Test)**
FiftyOne can assign tags to define splits.
```python
import fiftyone.utils.random as four

# Randomly assign tags 'train', 'val', 'test'
# This modifies the dataset in-place by adding tags
four.random_split(
    dataset,
    {"train": 0.7, "val": 0.2, "test": 0.1}
)

# Verify splits
print(dataset.count_sample_tags())
# {'train': 700, 'val': 200, 'test': 100, ...}
```

---

### Phase 2: Augmentation (Albumentations Integration)
The goal is to generate *new* augmented samples and add them to the dataset, essentially multiplying your training set.

**Strategy: Offline Augmentation via Plugin**
Instead of augmenting on-the-fly during training, we create physical copies. This is required if you want to inspect the data or require complex geometric transforms (like "Crop to Square") before the model sees them.

**Code Workflow:**
```python
import albumentations as A

# Define your augmentation pipeline
# Example: Random Crop to 640x640 and Horizontal Flip
transforms = A.Compose([
    A.RandomCrop(width=640, height=640),
    A.HorizontalFlip(p=0.5),
    # Add standardized resizing if needed for RF-DETR
    A.Resize(640, 640) 
])

# Use the plugin logic (conceptual wrapper, normally done via App or Python SDK)
# Note: The plugin provides an operator `augment_with_albumentations`.
# For programmable Python scripts without the GUI, utilize the plugin's internal functions 
# or write a direct loop using FiftyOne's API.

# RECOMMENDED SCRIPT APPROACH (More robust than Plugin GUI for pipelines):
samples_to_augment = dataset.match_tags("train") # Only augment training data!

augmented_samples = []
for sample in samples_to_augment:
    # 1. Load image
    img = cv2.imread(sample.filepath)
    h, w, _ = img.shape
    
    # 2. Get Detections (normalized coordinates)
    detections = sample.ground_truth.detections
    bboxes = [[d.bounding_box[0], d.bounding_box[1], d.bounding_box[2], d.bounding_box[3], d.label] for d in detections]
    
    # Albumentations expects [x_min, y_min, w, h] if format is 'coco' or similar
    # FiftyOne uses [x_min, y_min, w, h] normalized (0-1).
    
    # 3. Apply Transform
    transformed = transforms(image=img, bboxes=bboxes, class_labels=[b[-1] for b in bboxes])
    aug_img = transformed['image']
    aug_bboxes = transformed['bboxes']
    
    # 4. Create New Sample
    new_filepath = f"/path/to/save/aug_{sample.id}.jpg"
    cv2.imwrite(new_filepath, aug_img)
    
    new_detections = []
    for bbox, label in zip(aug_bboxes, [b[-1] for b in bboxes]):
        # bbox is [x, y, w, h] normalized? Check Albumentations config.
        # If A.Compose defaults, ensure `bbox_params` matches FiftyOne.
        new_detections.append(fo.Detection(label=label, bounding_box=bbox))
        
    new_sample = fo.Sample(filepath=new_filepath)
    new_sample["ground_truth"] = fo.Detections(detections=new_detections)
    new_sample.tags = ["train", "augmented"] # Mark as augmented
    augmented_samples.append(new_sample)

# Add all at once
dataset.add_samples(augmented_samples)
dataset.save()
```

**Crucial bbox_params for Albumentations + FiftyOne:**
FiftyOne stores boxes as `[x-top-left, y-top-left, width, height]` with values normalized to `[0, 1]`.
```python
transforms = A.Compose([
    ...
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'])) 
# 'coco' format in Albumentations is [x_min, y_min, width, height], NOT normalized.
# 'yolo' format in Albumentations is [x_center, y_center, width, height], normalized.
# 'albumentations' format is [x_min, y_min, x_max, y_max], normalized.
```
**Correction**: FiftyOne `[x, y, w, h]` normalized maps closest to `coco` (if unnormalized) or custom handling.
*Best Practice*: Convert FiftyOne bounds to pixel coordinates -> Apply Albumentations ('coco' format) -> Convert back to normalized -> Save to FiftyOne.

---

### Phase 3: Exporting for Training
You need a dataset structure compatible with **both** RF-DETR and YOLO.
*   **RF-DETR**: Uses standard COCO JSON format (`instances_train.json`).
*   **YOLO**: Uses a folder of images and a folder of `.txt` files with matching basenames.

**The "Hybrid Export" Strategy**
To save disk space and ensure consistency, export images *once* and generate both label formats.

1.  **Export to YOLO (Images + Labels)**
    YOLO requires a specific directory structure.
    ```python
    export_dir = "/path/to/export"
    
    # Export Training Split
    dataset.match_tags("train").export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="train",
        classes=["person", "car"] # Enforce class ordering
    )
    
    # Export Validation Split
    dataset.match_tags("val").export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="val"
    )
    ```
    *Result*: `images/train`, `labels/train`, `images/val`, `labels/val`, `dataset.yaml`.

2.  **Generate COCO JSON (RF-DETR)**
    We need a COCO JSON that points to the *existing* images exported in step 1.
    
    ```python
    # Re-export just the labels in COCO format, but point to the images we just created.
    # PRO TIP: FiftyOne's COCO exporter usually wants to write images. 
    # Workaround: Export to a temporary location with symlinks, or just generate the JSON.
    
    dataset.match_tags("train").export(
        export_dir=export_dir, # Same root
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
        labels_path="annotations/instances_train.json", # Custom location
        splt="train",
        export_media=False # CRITICAL: Do not re-export/copy images
    )
    # Note: When export_media=False, the JSON file names might refer to the original paths 
    # or just filenames. RF-DETR usually just needs the filename to match the file in folder.
    # Ensure the 'file_name' in JSON matches the actual files in `images/train`.
    ```

---

## 4. Key Considerations for Object Detection

1.  **Tag Management**: Always tag your original data `original` and augmented data `augmented`. This lets you easily ablating the impact of augmentation later: `view = dataset.match_tags(["original", "augmented"])`.
2.  **Field Selection**:
    *   `ground_truth`: Your manual annotations.
    *   `predictions`: Model outputs.
    *   **Decision**: for training, *always* export `ground_truth`. If you want to train on model predictions (knowledge distillation), duplicate the `predictions` field to `ground_truth` on a new view or clone of the dataset.
3.  **Validation**: Never apply geometric augmentations (crop, rotation) to your validation or test sets unless you specifically want to test robustness. Resize is acceptable, but usually handled by the model's preprocessing.
    *   *Rule*: `dataset.match_tags("train")` gets augmented. `dataset.match_tags(["val", "test"])` stays pure.

## 5. Alternatives

| Method | Pros | Cons |
| :--- | :--- | :--- |
| **FiftyOne Plugin (Interactive)** | Good for visual check. | Hard to automate in CI/CD pipeline. |
| **Python Script (Loop)** | Full control, reproducible. | Requires handling coord conv manually. |
| **Torchvision/Dataloader** | No disk space usage. | Can't visually debug easily; slower training (CPU bound). |

**Recommendation**: Use the **Python Script** approach to generate a static "v1_augmented" version of your dataset. This freezes your data state, ensuring that if you retrain RF-DETR and YOLO models later, they see the exact same pixels.

## YOLO Label Path Resolution & Zero-Copy Export Strategy (Research)

**Date:** 2026-02-11
**Status:** Complete
**Objective:** Investigate whether YOLOv11/YOLO26 support custom label directory paths in `dataset.yaml`, and design a zero-copy export strategy that ensures FiftyOne-exported clean labels are used during training (not original 6-column files).

---

### 1. Executive Summary

**No YOLO version (v5, v8, v11, YOLO26) supports custom label path overrides.** The label resolution is hardcoded via `img2label_paths()` which replaces the last `/images/` segment in image paths with `/labels/`. This is baked into `YOLODataset.get_labels()` and cannot be overridden via `dataset.yaml`.

**Solution**: Use FiftyOne's `export_media="symlink"` mode. This creates lightweight symlinks to original images inside `export_dir/images/{split}/` while generating fresh 5-column labels in `export_dir/labels/{split}/`. YOLO's path derivation naturally finds the exported labels. Zero disk bloat for images, clean labels for training.

### 2. Detailed Findings

#### 2.1 `img2label_paths()` is Hardcoded (No Override)

**Location**: `ultralytics/data/utils.py:L45-48`
```python
def img2label_paths(img_paths: list[str]) -> list[str]:
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]
```

**Called at**: `ultralytics/data/dataset.py:L165`
```python
def get_labels(self):
    self.label_files = img2label_paths(self.im_files)  # No override possible
```

This replaces the **last** occurrence of `/images/` with `/labels/` and changes the extension to `.txt`. There are no constructor parameters, YAML keys, or environment variables to override this behavior.

#### 2.2 `dataset.yaml` Path Resolution

**Location**: `ultralytics/data/utils.py:L386-477` (`check_det_dataset()`)

The YAML is parsed as follows:
1. `path:` becomes the dataset root directory
2. `train:`, `val:`, `test:` are resolved relative to `path:`
3. If the resolved path is a **directory**, images are globbed from it
4. If the resolved path is a **file** (`.txt`), image paths are read line-by-line

**Example YAML processing:**
```yaml
path: /data/exports/my_dataset
train: ./images/train/   # → glob images from /data/exports/my_dataset/images/train/
val: ./images/val/        # → glob images from /data/exports/my_dataset/images/val/
names: {0: 'cat', 1: 'dog'}
```

For each image at `/data/exports/my_dataset/images/train/img001.jpg`, YOLO looks for a label at `/data/exports/my_dataset/labels/train/img001.txt`.

#### 2.3 FiftyOne Label Filename Convention

**Finding**: FiftyOne uses `os.path.basename()` + `os.path.splitext()[0]` (stem) of the image path as the label filename UUID.

**Source**: `fiftyone/utils/data/exporters.py:L1231-1247` (`MediaExporter._get_uuid()`)
```python
def _get_uuid(self, path):
    if self.export_mode in (False, "manifest"):
        rel_dir = self.rel_dir
    else:
        rel_dir = self.export_path
    if rel_dir is not None:
        uuid = fou.safe_relpath(path, rel_dir)
    else:
        uuid = os.path.basename(path)
    if self.ignore_exts:
        uuid = os.path.splitext(uuid)[0]
    return uuid
```

With `ignore_exts=True` (set in YOLO exporter at L1040):
- Image `/original/dataset/images/train/photo_001.jpg` → UUID = `photo_001`
- Label written to `export_dir/labels/train/photo_001.txt`

**Confirmation**: Label `.txt` filenames match image filenames (stem). ✅

#### 2.4 The Zero-Copy Problem

When using `export_media=False` (the original zero-copy approach):
1. FiftyOne writes labels to `export_dir/labels/{split}/` ✅
2. FiftyOne **skips writing `dataset.yaml`** (L1065: `if self.export_media == False: return`) ❌
3. Our custom `setup_absolute_yaml()` writes a YAML where `train:` points to a `.txt` file listing absolute image paths
4. YOLO reads images from those absolute paths (e.g., `/data/raw/images/train/img001.jpg`)
5. `img2label_paths()` replaces `/images/` in the **original** path → looks for `/data/raw/labels/train/img001.txt`
6. **This finds the ORIGINAL label files** (possibly 6-column), not the FiftyOne-exported clean labels ❌

#### 2.5 The Symlink Solution

FiftyOne's YOLO exporter explicitly supports `export_media="symlink"`:

**Source**: `fiftyone/utils/yolo.py:L1038`
```python
self._media_exporter = foud.ImageExporter(
    self.export_media,
    export_path=self.data_path,
    supported_modes=(True, False, "move", "symlink"),  # ← symlink is supported!
    ...
)
```

**With `export_media="symlink"`:**
1. **Images**: Symlinks are created in `export_dir/images/{split}/` pointing to original image files (zero disk bloat)
2. **Labels**: Fresh 5-column `.txt` files are generated in `export_dir/labels/{split}/`
3. **YAML**: `dataset.yaml` IS written (L1065 check passes: `"symlink" != False`) with `train: ./images/train/`
4. **YOLO resolution**: Images found at `export_dir/images/train/img001.jpg` → labels found at `export_dir/labels/train/img001.txt` ✅+

**Resulting directory structure:**
```
export_dir/
├── dataset.yaml          # Auto-generated by FiftyOne
├── images/
│   ├── train/
│   │   ├── img001.jpg → /original/path/img001.jpg  (symlink)
│   │   └── img002.jpg → /original/path/img002.jpg  (symlink)
│   └── val/
│       └── img003.jpg → /original/path/img003.jpg  (symlink)
└── labels/
    ├── train/
    │   ├── img001.txt    (FiftyOne-generated, 5-column, clean)
    │   └── img002.txt    (FiftyOne-generated, 5-column, clean)
    └── val/
        └── img003.txt    (FiftyOne-generated, 5-column, clean)
```

### 3. Comparison of Export Modes

| Feature | `export_media=False` (old) | `export_media="symlink"` (new) | `export_media=True` (copy) |
|:---|:---|:---|:---|
| **Disk usage** | Labels only (~KB) | Labels + symlinks (~KB) | Labels + full images (GB) |
| **`dataset.yaml`** | NOT generated by FiftyOne | Auto-generated ✅ | Auto-generated ✅ |
| **Labels used by YOLO** | Original files (broken!) | FiftyOne-exported (clean) ✅ | FiftyOne-exported (clean) ✅ |
| **6-column safe?** | ❌ No | ✅ Yes | ✅ Yes |
| **Portability** | Paths break if original moves | Symlinks break if original moves | Fully self-contained |
| **Cross-platform** | Works everywhere | Symlinks may not work on Windows | Works everywhere |

### 4. Implementation Impact

**Change in `dataset_manager.py`**: Replace `export_media=False` with `export_media="symlink"` for the default zero-copy mode:
```python
# Old (broken):
split_view.export(..., export_media=copy_images)  # copy_images=False means export_media=False

# New (correct):
export_mode = True if copy_images else "symlink"
split_view.export(..., export_media=export_mode)
```

This eliminates the need for:
- Custom `setup_absolute_yaml()` function
- Custom `{split}.txt` listing files
- Manual YAML generation

FiftyOne handles everything natively with the symlink mode.
