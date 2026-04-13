"""Backend registry with explicit backend selection."""

from __future__ import annotations

from pathlib import Path

from hephaestus.backends.ardor.backend import ArdorBackend
from hephaestus.backends.base import ExecutionBackend
from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.backends.local_process_backend import LocalProcessBackend
from hephaestus.config_loader import ConfigError, load_config_file


def resolve_backend(name: str, config_dir: Path = Path("configs")) -> ExecutionBackend:
    if name == "dry_run":
        return DryRunBackend()
    if name == "local_process":
        return LocalProcessBackend()
    if name == "ardor":
        return ArdorBackend(config_dir=config_dir)
    raise ValueError(f"Unsupported backend: {name}")


def resolve_backend_from_system(config_dir: Path = Path("configs")) -> ExecutionBackend:
    system_cfg = load_config_file(config_dir / "system.yaml")
    backend_name = str(system_cfg.get("default_backend", "")).strip()
    if not backend_name:
        raise ConfigError("system config missing default_backend")
    return resolve_backend(backend_name, config_dir=config_dir)
