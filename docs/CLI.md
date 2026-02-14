# CLI Reference

This document describes the production CLI surface for `od_training`.

For module-level API documentation, see `docs/src/README.md`.

## 1. What `odt` Is

`odt` is the unified command-line interface for this repository. It provides one consistent entrypoint for:

- Dataset preparation, augmentation, conversion, export, and dataset viewing
- Model training (YOLO and RF-DETR)
- Model inference (YOLO and RF-DETR)
- Utility operations (environment verification, Roboflow download/upload)

## 2. Entrypoints

Supported entrypoints:

```bash
odt <group> <command> [args...]
```

Equivalent module entrypoint:

```bash
python -m od_training.cli.main <group> <command> [args...]
```

`group` values:

- `dataset`
- `train`
- `infer`
- `utility`

## 3. Dispatch Model

The top-level dispatcher is `src/od_training/cli/main.py`, and each command forwards to a module `main(argv)`.

| CLI Command | Implementation |
|---|---|
| `dataset manage` | `src/od_training/dataset/manager.py:main` |
| `dataset convert` | `src/od_training/dataset/convert_cli.py:main` |
| `dataset view` | `src/od_training/dataset/view.py:main` |
| `dataset augment-preview` | `src/od_training/dataset/augment_preview.py:main` |
| `train yolo` | `src/od_training/train/yolo.py:main` |
| `train rfdetr` | `src/od_training/train/rfdetr.py:main` |
| `infer run` | `src/od_training/infer/runner.py:main` |
| `utility verify-env` | `src/od_training/utility/verify_env.py:main` |
| `utility config-init` | `src/od_training/utility/config_cli.py:main_init` |
| `utility config-show` | `src/od_training/utility/config_cli.py:main_show` |
| `utility upload-weights` | `src/od_training/utility/roboflow_upload.py:main` |
| `utility download-roboflow` | `src/od_training/utility/roboflow_download.py:main` |

Unknown command behavior:

- If `group + command` is invalid, CLI prints available commands and exits with code `2`.

## 4. Command Reference

### 4.1 `dataset manage`

Purpose:

- Import/load a YOLO dataset in FiftyOne
- Optionally split samples into train/val/test tags
- Optionally augment samples
- Optionally export YOLO + COCO artifacts
- Optionally launch FiftyOne app

Syntax:

```bash
odt dataset manage [args...]
```

Arguments:

| Argument | Type | Default | Description |
|---|---:|---|---|
| `--dataset-dir` | str | `data/raw` | YOLO dataset directory or YAML path |
| `--name` | str | `my_dataset` | FiftyOne dataset name |
| `--split` | flag | off | Perform random split if split tags are missing |
| `--augment` | flag | off | Run augmentation |
| `--augment-tags` | list[str] | none | Restrict augmentation to samples with these tags |
| `--output-dataset` | str | none | Destination dataset for augmented samples |
| `--output-dir` | str | none | Output directory for augmented images |
| `--export-dir` | str | none | Directory to export YOLO + COCO artifacts |
| `--export-tags` | list[str] | none | Export only samples having all listed tags |
| `--copy-images` | flag | off | Copy media into export dir (default is symlink mode) |
| `--include-confidence` | flag | off | Export 6-column YOLO labels |
| `--label-field` | str | `ground_truth` | Detection field used for augmentation/export |
| `--classes-field` | str | auto | Field path for class discovery (`<label-field>.detections.label`) |
| `--view` | flag | off | Launch FiftyOne app |
| `--train-tag` | str | `train` | Train split tag |
| `--val-tag` | str | `val` | Validation split tag |
| `--test-tag` | str | `test` | Test split tag |

Key behavior:

- Default export mode is symlink-based (`export_media="symlink"`).
- COCO files are generated as `_annotations_<split>.coco.json`.
- COCO basename collisions are detected; full paths are kept to avoid corruption.
- Confidence-column export is opt-in via `--include-confidence` (6-column YOLO labels).
- Cross-repo handoff contract: `docs/context/dst_handoff_contract.md`.

### 4.2 `dataset convert`

Purpose:

- Convert between YOLO and COCO formats

Syntax:

```bash
odt dataset convert <mode> --input <path> --output <path> [--images <dir>]
```

Positional:

- `mode`: `yolo2coco` or `coco2yolo`

Arguments:

| Argument | Type | Required | Description |
|---|---:|---:|---|
| `--input` | str | yes | Input labels directory or COCO path |
| `--images` | str | for `yolo2coco` | Image directory used to resolve dimensions |
| `--output` | str | yes | Output file/directory |

Key behavior:

- `yolo2coco` exits with code `1` when `--images` is omitted.

### 4.3 `dataset view`

Purpose:

- Open a FiftyOne dataset by name
- Optionally import from a YOLO directory/YAML if the dataset does not already exist

Syntax:

```bash
odt dataset view <name> [--import-dir <path>]
```

Arguments:

| Argument | Type | Required | Description |
|---|---:|---:|---|
| `name` | str | yes | FiftyOne dataset name |
| `--import-dir` | str | no | YOLO dataset path to import if missing |

Exit behavior:

- Returns `1` if dataset is missing and `--import-dir` is not provided.

### 4.4 `dataset augment-preview`

Purpose:

- Load a YOLO dataset and generate a preview augmented sample
- Save preview to `runs/augmentation_test/aug_sample.jpg`

Syntax:

```bash
odt dataset augment-preview --dataset <path> [--name <dataset>] [--yaml <path>]
```

Arguments:

| Argument | Type | Required | Description |
|---|---:|---:|---|
| `--dataset` | str | yes | YOLO dataset directory |
| `--name` | str | no | FiftyOne dataset name (`my_dataset`) |
| `--yaml` | str | no | Explicit YAML path |

### 4.5 `train yolo`

Purpose:

- Train Ultralytics YOLO models with ClearML task initialization

Syntax:

```bash
odt train yolo [args...]
```

Arguments:

| Argument | Type | Default | Description |
|---|---:|---|---|
| `--model` | str | `yolo11n.pt` | Weights/model spec |
| `--data` | str | required | Dataset YAML path |
| `--project` | str | `YOLO_Training` | Output project/run root |
| `--name` | str | `exp` | Run name |
| `--epochs` | int | `100` | Epoch count |
| `--batch` | int | `16` | Batch size |
| `--imgsz` | int | `640` | Image size |
| `--device` | str | auto | Device override |
| `--no-validate-data` | flag | off | Skip dataset preflight validation |
| `--fail-on-validation-warnings` | flag | off | Treat preflight warnings as fatal |

Passthrough behavior:

- Unknown CLI args are parsed and forwarded to `model.train(**kwargs)`.

Special behavior:

- By default, runs preflight validation of YOLO split directories before training.
- If model name contains `yolo26` and optimizer is not provided, `optimizer="MuSGD"` is set.
- If device is non-CPU and epochs > 1, TensorRT export is attempted.

### 4.6 `train rfdetr`

Purpose:

- Train RF-DETR models on a COCO split directory contract (`train/`, `valid/`, `test/`)

Syntax:

```bash
odt train rfdetr [args...]
```

Arguments:

| Argument | Type | Default | Description |
|---|---:|---|---|
| `--dataset` | str | required | Dataset root containing COCO split subdirs |
| `--model` | str | `rf-detr-resnet50` | RF-DETR model key |
| `--epochs` | int | `50` | Epoch count |
| `--batch` | int | `8` | Batch size |
| `--lr` | float | `1e-4` | Learning rate |
| `--project` | str | `RF-DETR_Training` | Project name |
| `--name` | str | `exp` | Experiment name |
| `--device` | str | auto | Device override |
| `--no-validate-data` | flag | off | Skip dataset preflight validation |
| `--fail-on-validation-warnings` | flag | off | Treat preflight warnings as fatal |

Passthrough behavior:

- Unknown args are parsed and forwarded to `model.train(**kwargs)`.

Special behavior:

- By default, runs preflight validation of RF-DETR split contract before training.
- `tensorboard=True` and `wandb=True` are defaults unless overridden.
- If `checkpoint_best_regular.pth` is missing, training only fails if no other `.pth` checkpoint exists.

### 4.7 `infer run`

Purpose:

- Run inference with YOLO or RF-DETR on images, directories, videos, webcam (`0`), or RTSP source

Syntax:

```bash
odt infer run [args...]
```

Arguments:

| Argument | Type | Default | Description |
|---|---:|---|---|
| `--source` | str | required | Image/video/dir/`0`/RTSP source |
| `--model` | str | required | Model name or weights path |
| `--type` | enum | required | `yolo` or `rfdetr` |
| `--rfdetr-arch` | str | none | RF-DETR architecture hint for custom weights |
| `--conf` | float | `0.3` | Confidence threshold |
| `--iou` | float | `0.5` | IoU threshold for YOLO |
| `--show` | flag | off | Display OpenCV window |
| `--save-dir` | str | none | Save outputs if provided |

Output naming:

- Single image: `<stem>_annotated.jpg`
- Directory images: `<stem>_annotated.jpg` per input
- Video: `output_video.mp4`

### 4.8 `utility verify-env`

Purpose:

- Validate required runtime dependencies and accelerator availability

Syntax:

```bash
odt utility verify-env
```

Checks:

- Python runtime
- Critical deps: `numpy`, `pydantic`, `torch`, `ultralytics`
- Optional deps: `rfdetr`, `roboflow`
- Hardware: CUDA and MPS availability

Exit behavior:

- Exits `1` if critical packages are missing
- Exits `0` otherwise

### 4.9 `utility config-init`

Purpose:

- Create local runtime config template at resolved location (or explicit path)

Syntax:

```bash
odt utility config-init [--config /path/to/local_config.json]
```

Arguments:

| Argument | Type | Required | Description |
|---|---:|---:|---|
| `--config` | str | no | Optional config JSON path |

Behavior:

- Creates a template if missing; returns state `created` or `exists`.
- Prints JSON payload with resolved path.

### 4.10 `utility config-show`

Purpose:

- Show current resolved local config JSON (optionally masked)

Syntax:

```bash
odt utility config-show [--config /path/to/local_config.json] [--mask-secrets]
```

Arguments:

| Argument | Type | Required | Description |
|---|---:|---:|---|
| `--config` | str | no | Optional config JSON path |
| `--mask-secrets` | flag | no | Mask secret-like values in output |

Secret masking:

- Masks keys containing: `api_key`, `token`, `secret`, `password`.

### 4.11 `utility upload-weights`

Purpose:

- Upload a local checkpoint to Roboflow Deploy

Syntax:

```bash
odt utility upload-weights [args...]
```

Arguments:

| Argument | Type | Required | Description |
|---|---:|---:|---|
| `--config` | str | no | Optional config JSON path |
| `--api-key` | str | no | Roboflow API key (optional if configured) |
| `--workspace` | str | conditional | Workspace ID |
| `--project` | str | conditional | Project ID |
| `--version` | int | yes | Dataset version number |
| `--weights` | str | yes | Weights path |
| `--type` | str | no | Deploy model type (`yolov8`, `rf-detr`, etc.) |

Validation behavior:

- Fails if weights file does not exist
- Fails if weights file is empty
- Fails if required Roboflow settings are unresolved

### 4.12 `utility download-roboflow`

Purpose:

- Download a Roboflow dataset, optionally unzip and import into FiftyOne

Syntax:

```bash
odt utility download-roboflow [args...]
```

Arguments:

| Argument | Type | Required | Description |
|---|---:|---:|---|
| `--config` | str | no | Optional config JSON path |
| `--api-key` | str | no | Roboflow API key |
| `--workspace` | str | conditional | Workspace ID |
| `--project` | str | conditional | Project ID |
| `--version` | int | conditional | Dataset version |
| `--format` | str | conditional | Download format (`yolov11`, `coco`, etc.) |
| `--download-dir` | str | no | Download destination |
| `--unzip` | flag | default on | Unzip downloaded archive |
| `--no-unzip` | flag | no | Disable unzip |
| `--delete-zip` | flag | no | Remove archive after extraction |
| `--import-fiftyone` | flag | no | Import into FiftyOne |
| `--fiftyone-name` | str | no | FiftyOne dataset name |
| `--train-tag` | str | no (`train`) | Training tag |
| `--val-tag` | str | no (`val`) | Validation tag |
| `--test-tag` | str | no (`test`) | Test tag |
| `--interactive`, `-i` | flag | no | Prompt for missing values interactively |

Interactive behavior:

- Interactive prompts are used when:
  - `--interactive` is provided, or
  - `--version` or `--format` is missing.

## 5. Configuration Resolution

Runtime config file:

- `ODT_CONFIG_PATH` (if set)
- `config/local_config.json` (if present in repo)
- `~/.config/od_training/local_config.json`

Relevant keys:

- `roboflow.api_key`
- `roboflow.workspace`
- `roboflow.project`
- `data_dir`

Resolution priority:

1. CLI arguments
2. local config JSON
3. Environment variables (`ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT`)

Notes:

- Placeholder values (`<...>`) are treated as unset.
- CLI returns explicit errors when required values are unresolved.

## 6. Examples

Train YOLO:

```bash
odt train yolo \
  --model yolo11n.pt \
  --data runs/pipeline_test/dataset.yaml \
  --epochs 10 --batch 8 --imgsz 640
```

Train RF-DETR:

```bash
odt train rfdetr \
  --dataset data/my_coco \
  --model rfdetr_nano \
  --epochs 20 --batch 4 --lr 1e-4 \
  --device cuda:0 \
  --num-workers 8 --output-dir runs/rfdetr_exp1
```

Run inference:

```bash
odt infer run \
  --type yolo \
  --model runs/train/weights/best.pt \
  --source data/images \
  --conf 0.25 \
  --save-dir runs/infer
```

Download and import Roboflow data:

```bash
odt utility download-roboflow \
  --workspace my-workspace \
  --project my-project \
  --version 3 \
  --format yolov11 \
  --import-fiftyone \
  --fiftyone-name my_project_v3
```
