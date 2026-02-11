"""Local runtime configuration helpers.

This module manages a git-ignored local config file used for credentials and
other machine-specific settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


_DEFAULT_CONFIG: Dict[str, Any] = {
    "_instructions": {
        "note": "Fill placeholders in this file with your local credentials."
    },
    "roboflow": {
        "api_key": "<PASTE_ROBOFLOW_API_KEY_HERE>",
        "workspace": "<OPTIONAL_DEFAULT_WORKSPACE_ID>",
        "project": "<OPTIONAL_DEFAULT_PROJECT_ID>",
    },
    "data_dir": "data",
}


def get_repo_root() -> Path:
    """Return repository root path for the package layout."""
    # src/od_training/utility/runtime_config.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def get_config_path() -> Path:
    """Return absolute path to the local runtime config file."""
    return get_repo_root() / "config" / "local_config.json"


def ensure_local_config(config_path: Path | None = None) -> Tuple[Path, bool]:
    """Ensure local config exists, creating it with placeholders if needed.

    Returns:
        (config_path, created_flag)
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            json.dump(_DEFAULT_CONFIG, f, indent=2)
            f.write("\n")
        return path, True

    return path, False


def load_local_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load local runtime configuration JSON as a dictionary.

    Args:
        config_path: Optional explicit config file path.

    Returns:
        Parsed JSON mapping.
    """
    path, _ = ensure_local_config(config_path)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in local config: {path} ({e})")

    if not isinstance(data, dict):
        raise ValueError(f"Local config must be a JSON object: {path}")

    return data


def _is_placeholder(value: Any) -> bool:
    """Return whether a config value should be treated as unset placeholder."""
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    return stripped == "" or (stripped.startswith("<") and stripped.endswith(">"))


def get_roboflow_api_key(explicit_key: str | None = None) -> str:
    """Resolve Roboflow API key from CLI override, config, or environment.

    Args:
        explicit_key: Optional explicit API key from caller/CLI.

    Returns:
        Resolved API key string.
    """
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    # 1. Try local config
    cfg = load_local_config()
    value = cfg.get("roboflow", {}).get("api_key", "")
    
    if not _is_placeholder(value):
        return str(value).strip()

    # 2. Fallback to environment variable
    import os
    env_value = os.environ.get("ROBOFLOW_API_KEY")
    if env_value and env_value.strip():
        return env_value.strip()

    # 3. Fail
    path = get_config_path()
    raise ValueError(
        "Roboflow API key is not configured. "
        f"Set 'roboflow.api_key' in {path} or set ROBOFLOW_API_KEY env var."
    )


def get_roboflow_default(field: str) -> str | None:
    """Get optional roboflow default fields from local config.

    Supported fields: ``workspace``, ``project``.
    Analysis order:
      1. local_config.json
      2. Environment variables (ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT)
    """
    if field not in {"workspace", "project"}:
        raise ValueError(f"Unsupported roboflow field: {field}")

    # 1. Try local config
    cfg = load_local_config()
    value = cfg.get("roboflow", {}).get(field)
    if value and not _is_placeholder(value):
        return str(value).strip()

    # 2. Fallback to environment variable
    import os
    env_var_map = {
        "workspace": "ROBOFLOW_WORKSPACE",
        "project": "ROBOFLOW_PROJECT"
    }
    env_key = env_var_map[field]
    env_value = os.environ.get(env_key)
    
    if env_value and env_value.strip():
        return env_value.strip()
        
    return None


def get_data_dir() -> str:
    """Get data directory path from local config.
    
    Returns:
        Path to data directory (defaults to 'data').
    """
    cfg = load_local_config()
    value = cfg.get("data_dir")
    if value and not _is_placeholder(value):
        return str(value).strip()
    return "data"
