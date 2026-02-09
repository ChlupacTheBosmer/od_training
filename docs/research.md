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
*   **Optimizer**: Default `MuSGD` (Muon + SGD) handles deep transformer backbones better than vanilla SGD.
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

