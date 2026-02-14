"""Local runtime configuration helpers.

This module manages a git-ignored local config file used for credentials and
other machine-specific settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_CONFIG_ENV_VAR = "ODT_CONFIG_PATH"
DEFAULT_LOCAL_CONFIG_FILENAME = "local_config.json"
DEFAULT_CONFIG_SUBDIR = "od_training"


class RoboflowRuntimeConfig(BaseModel):
    """Typed roboflow config section."""

    model_config = ConfigDict(extra="allow")

    api_key: str = "<PASTE_ROBOFLOW_API_KEY_HERE>"
    workspace: str = "<OPTIONAL_DEFAULT_WORKSPACE_ID>"
    project: str = "<OPTIONAL_DEFAULT_PROJECT_ID>"


class LocalRuntimeConfig(BaseModel):
    """Typed root config model for local runtime config JSON."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    instructions: Dict[str, Any] = Field(
        default_factory=lambda: {
            "note": "Fill placeholders in this file with your local credentials."
        },
        alias="_instructions",
    )
    roboflow: RoboflowRuntimeConfig = Field(default_factory=RoboflowRuntimeConfig)
    data_dir: str = "data"


_DEFAULT_CONFIG: Dict[str, Any] = LocalRuntimeConfig().model_dump(by_alias=True)


def get_repo_root() -> Path:
    """Return repository root path for editable `src` package layout."""
    return Path(__file__).resolve().parents[3]


def resolve_default_local_config_path() -> Path:
    """Resolve default local config path in a portable way.

    Resolution order:
    1. ``ODT_CONFIG_PATH`` environment variable
    2. repo-local ``config/local_config.json`` if present
    3. ``$XDG_CONFIG_HOME/od_training/local_config.json``
       (or ``~/.config/od_training/local_config.json``)
    """
    env_path = os.getenv(DEFAULT_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()

    repo_candidate = get_repo_root() / "config" / DEFAULT_LOCAL_CONFIG_FILENAME
    if repo_candidate.exists():
        return repo_candidate

    xdg_config_home = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    return xdg_config_home / DEFAULT_CONFIG_SUBDIR / DEFAULT_LOCAL_CONFIG_FILENAME


def get_config_path(config_path: Path | None = None) -> Path:
    """Return absolute path to the local runtime config file."""
    if config_path is not None:
        return Path(config_path).expanduser()
    return resolve_default_local_config_path()


def ensure_local_config(config_path: Path | None = None) -> Tuple[Path, bool]:
    """Ensure local config exists, creating it with placeholders if needed.

    Returns:
        (config_path, created_flag)
    """
    path = get_config_path(config_path)
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
    path, _ = ensure_local_config(config_path=config_path)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in local config: {path} ({e})")

    if not isinstance(data, dict):
        raise ValueError(f"Local config must be a JSON object: {path}")

    return data


def parse_local_config(raw: Dict[str, Any]) -> LocalRuntimeConfig:
    """Parse/validate untyped config dict into a typed config object."""
    try:
        return LocalRuntimeConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid local config schema: {e}") from e


def _is_placeholder(value: Any) -> bool:
    """Return whether a config value should be treated as unset placeholder."""
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    return stripped == "" or (stripped.startswith("<") and stripped.endswith(">"))


def get_roboflow_api_key(explicit_key: str | None = None, config_path: Path | None = None) -> str:
    """Resolve Roboflow API key from CLI override, config, or environment.

    Args:
        explicit_key: Optional explicit API key from caller/CLI.

    Returns:
        Resolved API key string.
    """
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    # 1. Try local config
    cfg = parse_local_config(load_local_config(config_path=config_path))
    value = cfg.roboflow.api_key
    
    if not _is_placeholder(value):
        return str(value).strip()

    # 2. Fallback to environment variable
    env_value = os.environ.get("ROBOFLOW_API_KEY")
    if env_value and env_value.strip():
        return env_value.strip()

    # 3. Fail
    path = get_config_path(config_path=config_path)
    raise ValueError(
        "Roboflow API key is not configured. "
        f"Set 'roboflow.api_key' in {path} or set ROBOFLOW_API_KEY env var."
    )


def get_roboflow_default(field: str, config_path: Path | None = None) -> str | None:
    """Get optional roboflow default fields from local config.

    Supported fields: ``workspace``, ``project``.
    Analysis order:
      1. local_config.json
      2. Environment variables (ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT)
    """
    if field not in {"workspace", "project"}:
        raise ValueError(f"Unsupported roboflow field: {field}")

    # 1. Try local config
    cfg = parse_local_config(load_local_config(config_path=config_path))
    value = getattr(cfg.roboflow, field)
    if value and not _is_placeholder(value):
        return str(value).strip()

    # 2. Fallback to environment variable
    env_var_map = {
        "workspace": "ROBOFLOW_WORKSPACE",
        "project": "ROBOFLOW_PROJECT"
    }
    env_key = env_var_map[field]
    env_value = os.environ.get(env_key)
    
    if env_value and env_value.strip():
        return env_value.strip()
        
    return None


def get_data_dir(config_path: Path | None = None) -> str:
    """Get data directory path from local config.
    
    Returns:
        Path to data directory (defaults to 'data').
    """
    cfg = parse_local_config(load_local_config(config_path=config_path))
    value = cfg.data_dir
    if value and not _is_placeholder(value):
        return str(value).strip()
    return "data"
