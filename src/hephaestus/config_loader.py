from __future__ import annotations

import json
from pathlib import Path


class ConfigError(ValueError):
    pass


def load_config_file(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"malformed config file: {path}: expected JSON-compatible content in .yaml file ({exc.msg})") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"config root must be an object: {path}")
    return payload


def load_named_config(config_dir: Path, subdir: str, name: str) -> dict[str, object]:
    path = config_dir / subdir / f"{name}.yaml"
    return load_config_file(path)
