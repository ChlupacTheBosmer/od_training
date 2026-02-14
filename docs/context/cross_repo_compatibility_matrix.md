# Cross-Repo Compatibility Matrix (`dst` + `od_training`)

Pinned/ranged baseline for a single shared environment across both repos.

## Core Runtime

| Package | Constraint | Rationale |
|---|---|---|
| `python` | `>=3.10` | Common baseline for both codebases |
| `numpy` | `>=1.23.5,<2.0.0` | Ultralytics/RF-DETR compatibility |
| `scipy` | `<1.13.0` | Known compatibility guardrail in training stack |
| `pydantic` | `>=1.10,<3.0.0` | Avoids v2/v3 breakage across tooling |

## Dataset/Curation Layer

| Package | Constraint | Rationale |
|---|---|---|
| `fiftyone` | `>=1.11,<2.0` | Shared dataset DB/API compatibility |
| `fiftyone-brain` | `>=0.21,<1.0` | Needed by `dst` brain/metric features |
| `label-studio-sdk` | `==1.0.20` | `dst` Label Studio integration path with NumPy `<2` |

## Training Layer

| Package | Constraint | Rationale |
|---|---|---|
| `torch` | `==2.7.1` | shared stable baseline across repos |
| `ultralytics` | `>=8.0` | YOLO train/infer |
| `rfdetr` | `rfdetr[metrics,plus]` | RF-DETR train/infer + metrics |
| `roboflow` | `>=1.2.13,<1.3` | avoids old `pyparsing==2.4.7` pin |
| `pyparsing` | `>=3,<4` | satisfies modern `matplotlib` required by FiftyOne |

## Operational Notes

- Keep one shared venv for both repos when working in `external/`.
- Install both repos editable into the same venv:
  - `pip install -e external/dst`
  - `pip install -e external/od_training`
- Re-run:
  - `dst --help`
  - `odt utility verify-env`
  - `python -m pip check`
