# Discovery Phase

## Questions

### Date: 2026-02-09

1.  **YOLO Version**:
    *   **User Answer**: Predominantly YOLOv11 and "YOLO26" (latest Ultralytics).
    *   **Action**: Verify "YOLO26" (likely typo or very new). Use latest `ultralytics` package.
2.  **RF-DETR Implementation**:
    *   **User Answer**: Roboflow RF-DETR.
    *   **Sources**:
        *   https://roboflow.com/model/rf-detr
        *   https://github.com/roboflow/rf-detr
    *   **Notes**: Use standalone but support Roboflow platform interaction (upload/download/dataset SDK).
3.  **Data Formats**:
    *   **User Answer**: Unknown, needs research.
4.  **Experiment Tracking**:
    *   **User Answer**: ClearML (previously used), or industry standard.
    *   **Action**: Will likely stick to ClearML or suggest W&B if better verification is needed.
5.  **Environment & Containerization**:
    *   **User Answer**: No Dockerfile. Use `requirements.txt`.
    *   **Context**: User has a base image; we install into persistent venv on Kubernetes PVC.
6.  **Dependency Management**:
    *   **User Answer**: `requirements.txt`.
7.  **Hardware & Distribution**:
    *   **User Answer**: Single GPU.

