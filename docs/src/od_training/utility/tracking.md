# Module: `src.od_training.utility.tracking`

- File: `src/od_training/utility/tracking.py`

## Purpose

Provides safe ClearML task initialization with graceful fallback.

## Functions

### `init_clearml_task(project_name: str, task_name: str, logger: Optional[Any] = None)`

Attempts to import and initialize a ClearML task.

Returns:

- `Task` instance when successful
- `None` when ClearML import/init fails

If `logger` is provided, failures are logged as warnings.
