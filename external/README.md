# External Reference Repositories

This directory is for **local reference clones only**.

- These repos are not runtime dependencies of this project.
- Do not import code from `external/` in project modules or tests.
- Use installed packages in `venv` for execution.

## Hard Guardrails

- Treat repositories under `external/` as read-only references for inspection.
- Refresh references before deep debugging or behavior comparison:
  - `bash scripts/sync_external_refs.sh --pull`
- When citing upstream behavior in notes/docs, include repo URL and commit SHA.

## Add/Update Repositories

1. Add one `git clone ...` command per line under **Clone commands**.
2. Keep each command in the form:
   - `git clone <repo-url>`
3. Repository directory name is inferred from URL basename (without `.git`).

## Sync Script

Use the helper script from repo root:

- Clone missing refs:
  - `bash scripts/sync_external_refs.sh`
- Also pull existing refs:
  - `bash scripts/sync_external_refs.sh --pull`
- Also disable push URL on remotes (safer for reference repos):
  - `bash scripts/sync_external_refs.sh --disable-push`

## Clone commands

git clone https://github.com/roboflow/rf-detr
git clone https://github.com/roboflow/roboflow-python
git clone https://github.com/roboflow/supervision
git clone https://github.com/ultralytics/ultralytics
