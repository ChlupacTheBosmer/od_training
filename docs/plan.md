# Implementation Plan

See the detailed implementation plan artifact at:
- [implementation_plan.md](file:///Users/chlup/.gemini/antigravity/brain/19f9af3c-1dc7-4e56-9898-1bc4e29fac5d/implementation_plan.md)

## Summary

All 5 phases of the audit fix have been implemented:

1. **Phase 2 – Critical**: Symlink export in `dataset_manager.py`, RFDETR_MAP NameError fix, checkpoint error verification
2. **Phase 3 – High**: Shared `device_utils.py` + `cli_utils.py`, integrated into both training scripts, `converters.py` rewrite with ValidationReport, `verify_env.py` exit codes
3. **Phase 4 – Medium**: `requirements.txt` pins, inference UX (FPS, filenames, save-dir), `upload_weights.py` (env var, preflight), `verify_pipeline.sh` symlink tests
4. **Phase 5 – Low**: Removed unused imports, hardened sys.path with `Path(__file__).resolve()`

## Phase 6: Feature Enhancements
- [x] Allow custom output directory for augmented images in `dataset_manager.py`
