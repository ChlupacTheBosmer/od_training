"""Experiment tracking helpers."""

from typing import Any, Optional


def init_clearml_task(project_name: str, task_name: str, logger: Optional[Any] = None):
    """Initialize a ClearML task if available.

    Returns:
        Task instance or ``None`` when ClearML is unavailable/unconfigured.
    """
    try:
        from clearml import Task
    except Exception as e:  # pragma: no cover - import environment dependent
        if logger:
            logger.warning("ClearML import failed: %s. Proceeding without tracking.", e)
        return None

    try:
        return Task.init(project_name=project_name, task_name=task_name, output_uri=True)
    except Exception as e:
        if logger:
            logger.warning("ClearML initialization failed: %s. Proceeding without tracking.", e)
        return None
