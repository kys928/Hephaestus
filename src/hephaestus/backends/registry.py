"""Backend registry with explicit backend selection."""

from __future__ import annotations

from hephaestus.backends.base import ExecutionBackend
from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.backends.local_process_backend import LocalProcessBackend


def resolve_backend(name: str) -> ExecutionBackend:
    if name == "dry_run":
        return DryRunBackend()
    if name == "local_process":
        return LocalProcessBackend()
    raise ValueError(f"Unsupported backend: {name}")
