"""CLI helpers for local runtime config management."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .runtime_config import ensure_local_config, load_local_config

_MASK_TOKEN = "***MASKED***"
_SECRET_KEY_HINTS = ("api_key", "token", "secret", "password", "passwd")


def _is_secret_key(key: str) -> bool:
    """Return whether ``key`` should be treated as secret-like."""
    lowered = key.lower()
    return any(hint in lowered for hint in _SECRET_KEY_HINTS)


def _mask_secrets(data: Any) -> Any:
    """Return deep-copied config structure with secret values redacted."""
    if isinstance(data, dict):
        masked: dict[str, Any] = {}
        for key, value in data.items():
            if _is_secret_key(str(key)):
                masked[str(key)] = _MASK_TOKEN
            else:
                masked[str(key)] = _mask_secrets(value)
        return masked
    if isinstance(data, list):
        return [_mask_secrets(v) for v in data]
    return data


def build_parser() -> argparse.ArgumentParser:
    """Build parser for ``odt utility config-*`` commands."""
    parser = argparse.ArgumentParser(description="Runtime config utilities")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to local config JSON (defaults to ODT_CONFIG_PATH/repo/XDG resolution).",
    )
    parser.add_argument(
        "--mask-secrets",
        action="store_true",
        help="Mask secret-like values in output (api_key/token/secret/password).",
    )
    return parser


def main(argv=None, mode: str = "show") -> int:
    """Execute config utility command.

    Args:
        argv: Optional argument list.
        mode: Command mode: ``init`` or ``show``.
    """
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser() if args.config else None

    if mode == "init":
        path, created = ensure_local_config(config_path=config_path)
        state = "created" if created else "exists"
        print(json.dumps({"ok": True, "path": str(path), "state": state}, indent=2))
        return 0

    if mode == "show":
        cfg = load_local_config(config_path=config_path)
        if args.mask_secrets:
            cfg = _mask_secrets(cfg)
        print(json.dumps(cfg, indent=2))
        return 0

    raise ValueError(f"Unsupported config utility mode: {mode}")


def main_init(argv=None) -> int:
    """CLI entrypoint for ``odt utility config-init``."""
    return main(argv=argv, mode="init")


def main_show(argv=None) -> int:
    """CLI entrypoint for ``odt utility config-show``."""
    return main(argv=argv, mode="show")

