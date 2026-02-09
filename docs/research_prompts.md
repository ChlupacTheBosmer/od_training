# Research Prompts for Parallel Agents

Use the following prompts to instruct separate agents. They will report back and populate `docs/research.md`.

## Research Area 1: RF-DETR Standalone & Reproducibility (Priority: High)
**Objective**: Determine how to train, validate, and test RF-DETR *locally* and on *Kubernetes* without relying on the Roboflow web platform for execution.

**Prompt**:
> You are an expert AI researcher. Your goal is to document the exact technical workflow for training **RF-DETR** (Roboflow DETR) in a standalone environment (local Linux/Mac or Kubernetes), **avoiding** reliance on the Roboflow web UI for triggering training.
>
> Please research and document the following in `docs/research.md` (append to existing content):
> 1.  **Installation**: What is the exact `pip install` command? Does it require specific PyTorch or CUDA versions different from standard YOLO?
> 2.  **Training Script**: Provide a **complete, working Python script** example to train RF-DETR on a local dataset.
>     *   How do we point it to a local COCO JSON file?
>     *   How do we configure hyperparameters (epochs, learning rate, batch size) via Python arguments?
> 3.  **Evaluation**: How do we run validation/testing on a separate test set after training?
> 4.  **Logging**: How do we integrate **ClearML** with the RF-DETR training loop? Does the `rfdetr` package expose a callback system?
> 5.  **Artifacts**: Where are weights saved? What is the format (`.pt`, `.pth`)?

---

## Research Area 2: YOLOv11 & YOLO26 Technical Specifics (Priority: High)
**Objective**: Clarify the differences and implementation details between YOLOv11 and the newly released "YOLO26".

**Prompt**:
> You are an expert AI researcher. Your goal is to clarify the usage of **YOLOv11** and **YOLO26** using the `ultralytics` package.
>
> Please research and document the following in `docs/research.md` (append to existing content):
> 1.  **Model Loading**: Confirm the exact model names for initialization (e.g., `YOLO("yolo11n.pt")` vs `YOLO("yolo26n.pt")`).
> 2.  **YOLO26 Specifics**: What makes YOLO26 different in terms of hyperparameters? Are there specific flags (e.g., related to NMS-free training) we must enable?
> 3.  **Hardware Optimization**: Since we are running on Kubernetes (NVIDIA GPU), are there specific export formats or distinct arguments recommended for YOLO26?
> 4.  **Metrics**: Does YOLO26 introduce new metrics or logging keys compared to v8/v11?

---

## Research Area 3: Data Interchange & Verification (Priority: Medium)
**Objective**: Design a robust workflow for converting between YOLO (txt) and COCO (json) formats, and validating dataset integrity.

**Prompt**:
> You are a Data Engineering expert. Your task is to provide solution for robust conversion between **YOLO** (Darknet txt) and **COCO** (JSON) formats.
>
> Please research and document the following in `docs/research.md`:
> 1.  **Conversion Tools**: Recommend a Python library or provide a **robust script** to convert:
>     *   YOLO -> COCO (Handling `train/val` splits correctly).
>     *   COCO -> YOLO.
> 2.  **Validation**: How can we efficiently validate a dataset before training (e.g., check for corrupt images, missing labels, negative bounding boxes)?
> 3.  **Roboflow SDK**: Provide a snippet to download a dataset from Roboflow in *both* formats (to use as ground truth for testing our converters).

---

## Research Area 4: RF-DETR Logging & Checkpoints (Follow-up) (Priority: Critical)
**Objective**: Close the gap on "black-box" training. We are willing to use ANY logging service (W&B, TensorBoard, CSV) if it is supported natively. We are not locked to ClearML.

**Prompt**:
> You are a Deep Learning Engineer. We need to monitor training progress for `rfdetr`.
>
> Please research and document the following in `docs/research.md` (append to RF-DETR section):
> 1.  **Native Logging**: Does `rfdetr` support any logging services out-of-the-box (e.g., Weights & Biases, TensorBoard, CSV)? How do we enable them?
> 2.  **Custom Integration**: If NO native logging exists, does the `Trainer` support a `callbacks` argument or expose logs so we can wire up our own logger (ClearML or otherwise)?
> 3.  **Checkpointing**: Where specifically does it save `best.pt` and `last.pt`? Is it configurable via `project` and `name` arguments like YOLO?
> 4.  **Validation Loop**: Does `model.val()` exist? If not, how do we run a validation pass on the *entire* validation set and get the mAP50/mAP50-95 score programmatically (not just printing it)?

---

## Research Area 5: Unified Environment Compatibility (Follow-up) (Priority: High)
**Objective**: Ensure `rfdetr` and `ultralytics` can coexist in one `requirements.txt` and venv without version hell.

**Prompt**:
> You are a DevOps Engineer. We need to install both `rfdetr` and `ultralytics` (latest) in the same Python environment on a Kubernetes pod.
>
> Please research and document in `docs/research.md`:
> 1.  **Dependency Conflict Check**: `rfdetr` requires strict environment. `ultralytics` is more flexible.
>     *   Check if `rfdetr` forces a specific `torch` or `numpy` version that breaks `ultralytics`.
>     *   Check `pydantic` versions (common conflict source).
> 2.  **Docker/Base Image**: The user has a generic base image. Should we install `rfdetr` *first* or *last*?
> 3.  **Python Version**: Confirmed **3.10** is best for `rfdetr`. Is `ultralytics` fully performant on 3.10? (Likely yes, but verify).

---

## Research Area 6: YOLO26 Fine-tuning Strategy (Follow-up) (Priority: Medium)
**Objective**: We know *what* YOLO26 is, now we need to know *how* to fine-tune it best for custom data.

**Prompt**:
> You are a YOLO Expert. We are training **YOLO26** on custom data.
>
> Please research and document in `docs/research.md`:
> 1.  **Freezing Layers**: How do we freeze the transformer backbone in YOLO26? Is there a `freeze=XX` argument valid for the new architecture?
> 2.  **Hyperparameters**: Does the `MuSGD` optimizer require different learning rates than standard AdamW? What is the recommended `lr0` for fine-tuning?
> 3.  **Data Augmentation**: Does the end-to-end head require different augmentation strategies (e.g., less Mosaic)?
