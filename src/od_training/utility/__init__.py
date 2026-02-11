"""Utility package for runtime config, CLI parsing, device, and integrations."""

from .cli import parse_unknown_args
from .device import resolve_device
from .runtime_config import (
    ensure_local_config,
    get_config_path,
    get_data_dir,
    get_repo_root,
    get_roboflow_api_key,
    get_roboflow_default,
    load_local_config,
)
from .tracking import init_clearml_task

__all__ = [
    "parse_unknown_args",
    "resolve_device",
    "ensure_local_config",
    "get_config_path",
    "get_data_dir",
    "get_repo_root",
    "get_roboflow_api_key",
    "get_roboflow_default",
    "load_local_config",
    "init_clearml_task",
]
