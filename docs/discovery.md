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

### Date: 2026-02-11 — Audit Fix Planning

8.  **Typed Config Model (Pydantic)**:
    *   **Question**: Should this be done in same pass as bug fixes, or as a follow-up phase?
    *   **User Answer**: Follow-up phase after audit fixes. Will revisit once done.
9.  **Dataset Contracts**:
    *   **Question**: Should validation block training automatically, or remain opt-in?
    *   **User Answer**: Implement validation logic now (opt-in), full enforcement later alongside typed config.
10. **Module Restructure**:
    *   **Question**: Do the `data_io`/`augment`/`export` module split now or defer?
    *   **User Answer**: Targeted changes now. Confirm they don't break anything, then move to next phase.
11. **Dependency Pinning Strategy**:
    *   **Question**: Exact versions or compatible ranges?
    *   **User Answer**: Widest compatible ranges. No exact version pins. Flexibility is key.
12. **copy_images=True Export**:
    *   **Question**: What should copy_images mode produce?
    *   **User Answer**: A fully self-contained dataset: copy images+labels into target dir, create dataset config files. Ready for upload elsewhere. Opt-in because it duplicates disk usage.
13. **Absolute Paths in data.yaml**:
    *   **Question**: Use relative paths in generated data.yaml?
    *   **User Answer**: NO — keep absolute paths. Dataset must resolve correctly when called from anywhere, not just the repo root. This is critical for Kubernetes pods/jobs.
14. **6-Column YOLO Labels (confidence scores)**:
    *   **Question**: Labels often have 6 columns (class xc yc w h conf) from model predictions. How to handle?
    *   **User Answer**: Keep confidence in annotation files for dataset curation. If training can't use 6-column, provide a helper to deal with it. Preferred: keep conf score in files.
    *   **Resolution**: FiftyOne handles this natively. On import, it reads the 6th column as `confidence` into `Detection.confidence`. On export via `YOLOAnnotationWriter`, it writes 5-column by default (training-ready). Pass `include_confidence=True` to preserve the 6th column. **No `strip_confidence` utility needed** — the FiftyOne pipeline already decouples on-disk format from export format.

15. **verify_pipeline.sh Changes**:
    *   **User Answer**: Don't implement configurable venv path. Make "clean" the default behavior with option to skip (`--no-clean`). This is a temporary test script.
16. **Logging Style**:
    *   **User Answer**: Keep bare `print()` everywhere. Don't standardize on `logging` module.
17. **sys.path Hacks / pyproject.toml**:
    *   **User Answer**: Don't use pyproject.toml or editable install. Don't require running from project root. Scripts must work from any working directory (Kubernetes pods). Keep `sys.path` approach but make it robust cross-platform.
18. **Shared CUDA Detection Logic**:
    *   **User Answer**: YOLO and RF-DETR should have analogous logic. Reusable code (like CUDA detection) should be extracted into a shared module.
19. **Zero-Copy Export Strategy (`export_media="symlink"`)**: (2026-02-11)
    *   **Question**: How should zero-copy export link FiftyOne-generated clean labels to original images?
    *   **Finding**: YOLO has no custom label path override — `img2label_paths()` is hardcoded to replace `/images/` → `/labels/`. Using `export_media=False` causes YOLO to read the **original** label files (possibly 6-column), not FiftyOne-exported ones.
    *   **Solution**: Use `export_media="symlink"` — FiftyOne creates symlinks in `images/{split}/` → original files, generates clean 5-col labels in `labels/{split}/`, and auto-writes `dataset.yaml`. YOLO's path resolution naturally finds the exported labels.
    *   **User Decision**: Approved. `"symlink"` is the new default zero-copy mode. `copy_images=True` (`export_media=True`) kept for portability.
20. **Symlink Portability (Kubernetes PVC)**: (2026-02-11)
    *   **Question**: Symlinks are absolute. Will they work across pods with different mount points?
    *   **Finding**: Absolute symlinks require consistent mount points. Relative symlinks would fix this but only if export + images are on the same PVC.
    *   **User Decision**: Export and images will be on **separate PVCs**. Keep absolute symlinks. **Enforce consistent mount points** across all pods — simpler than relative symlinks.

