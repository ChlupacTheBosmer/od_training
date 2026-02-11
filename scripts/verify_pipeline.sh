#!/bin/bash
# verify_pipeline.sh — End-to-end pipeline smoke test.
#
# Default: cleans up test artifacts before and after.
# Use --no-clean to keep artifacts for manual inspection.
# Use --no-clean to keep artifacts for manual inspection.
# set -e (Removed to allow manual error handling in checks)

NO_CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --no-clean) NO_CLEAN=true ;;
    esac
done

EXPORT_DIR="runs/pipeline_test"
FO_DATASET="dummy_manager_test"
FO_AUG_DATASET="dummy_manager_separate"

# ─── 0. Clean previous runs ─────────────────────────────────────────────────
cleanup() {
    echo "Cleaning test artifacts…"
    rm -rf "$EXPORT_DIR"
    rm -rf "data/augmented/$FO_DATASET"
    rm -rf "data/augmented/$FO_AUG_DATASET"
    python -c "
import fiftyone as fo
for name in ['$FO_DATASET', '$FO_AUG_DATASET']:
    if name in fo.list_datasets():
        fo.delete_dataset(name)
"
}

cleanup

# ─── 1. Import + split + augment to separate dataset ────────────────────────
echo ""
echo "=== Step 1: Import, Split & Augment ==="
if [ -f "venv/bin/activate" ]; then
    echo "Sourcing venv/bin/activate..."
    source venv/bin/activate
else
    echo "No venv found, assuming environment is already set up."
fi

python scripts/dataset_manager.py \
    --dataset-dir data/dummy_yolo/data.yaml \
    --name "$FO_DATASET" \
    --split \
    --augment \
    --augment-tags train \
    --output-dataset "$FO_AUG_DATASET"

# ─── 2. Export — Symlink mode (default, zero-copy) ──────────────────────────
echo ""
echo "=== Step 2: Export (symlink mode) ==="
python scripts/dataset_manager.py \
    --name "$FO_AUG_DATASET" \
    --export-dir "$EXPORT_DIR" \
    --export-tags augmented



# ─── 3. Verify symlink export structure ─────────────────────────────────────
echo ""
echo "=== Step 3: Verification ==="
PASS=0
FAIL=0

check() {
    local label="$1"
    local condition="$2"
    # Run condition in subshell, capture exit code, prevent set -e from killing script
    if eval "$condition"; then
        echo "  ✓ $label"
        ((PASS++))
    else
        echo "  ✗ $label"
        ((FAIL++))
        # Print debug info if failed
        echo "    Debug: Command failed: $condition"
    fi
}

# dataset.yaml
check "dataset.yaml exists" "[ -f '$EXPORT_DIR/dataset.yaml' ]"

# Symlinked images directory
check "images/train/ directory exists" "[ -d '$EXPORT_DIR/images/train' ]"

# Check for symlinks explicitly
# We use a loop to check if any file is a symlink
echo "  Checking for symlinks in $EXPORT_DIR/images/train..."
SYMLINK_COUNT=$(find "$EXPORT_DIR/images/train" -type l | wc -l)
if [ "$SYMLINK_COUNT" -gt 0 ]; then
    echo "  ✓ images/train/ contains $SYMLINK_COUNT symlinks"
    ((PASS++))
else
    echo "  ✗ images/train/ contains 0 symlinks (Expected >0)"
    ls -la "$EXPORT_DIR/images/train"
    ((FAIL++))
fi

# Generated labels
check "labels/train/ directory exists" "[ -d '$EXPORT_DIR/labels/train' ]"
check "labels/train/ contains .txt files" \
    "ls '$EXPORT_DIR/labels/train/'*.txt 1>/dev/null 2>&1"

# Labels are 5-column (no confidence)
check "Labels are 5-column (standard YOLO)" \
    "awk 'NF != 5 {exit 1}' '$EXPORT_DIR/labels/train/'*.txt"

# No old-style .txt listing files
check "No train.txt listing file (old mode removed)" "[ ! -f '$EXPORT_DIR/train.txt' ]"

# COCO JSON
check "COCO JSON exists" "[ -f '$EXPORT_DIR/_annotations_train.coco.json' ]"

# ─── 4. Report ──────────────────────────────────────────────────────────────
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

# ─── 5. Cleanup (unless --no-clean) ─────────────────────────────────────────
if [ "$NO_CLEAN" = false ]; then
    cleanup
else
    echo "Keeping test artifacts at: $EXPORT_DIR"
fi

if [ "$FAIL" -gt 0 ]; then
    echo "Pipeline verification FAILED."
    exit 1
fi

echo "Pipeline verification PASSED ✓"
