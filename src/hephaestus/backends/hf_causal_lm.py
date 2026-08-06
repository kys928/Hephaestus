"""Optional capability and identity helpers for Transformers causal-LM training.

This module deliberately imports no ML framework at module import time.  Core
Hephaestus remains usable when the training extra is not installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

REQUIRED_HF_TRAINING_PACKAGES = ("torch", "transformers", "tokenizers")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_manifest(path: str | Path) -> dict[str, str]:
    """Return stable component hashes for a local model/tokenizer directory."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"local model/tokenizer directory does not exist: {root}")
    entries: dict[str, str] = {}
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and not candidate.name.endswith(".partial"):
            entries[candidate.relative_to(root).as_posix()] = _hash_file(candidate)
    if not entries:
        raise ValueError(f"local model/tokenizer directory is empty: {root}")
    return entries


def directory_content_identity(path: str | Path) -> str:
    """Compute an immutable identity from every regular file in a directory."""

    encoded = json.dumps(directory_manifest(path), sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def installed_framework_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (*REQUIRED_HF_TRAINING_PACKAGES, "accelerate", "peft"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


@dataclass(frozen=True, slots=True)
class TransformersTrainingCapability:
    supported: bool
    missing_packages: tuple[str, ...] = ()
    framework_versions: dict[str, str | None] = field(default_factory=dict)
    devices: tuple[str, ...] = ("cpu",)
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "missing_packages": list(self.missing_packages),
            "framework_versions": dict(self.framework_versions),
            "devices": list(self.devices),
            "limitations": list(self.limitations),
        }


def transformers_training_capability() -> TransformersTrainingCapability:
    """Inspect the optional training extra without importing it."""

    missing = tuple(
        package
        for package in REQUIRED_HF_TRAINING_PACKAGES
        if importlib.util.find_spec(package) is None
    )
    limitations = (
        "Transformers training is an optional capability and is not imported by core Hephaestus.",
        "A tiny smoke run does not establish useful large-model training quality.",
    )
    return TransformersTrainingCapability(
        supported=not missing,
        missing_packages=missing,
        framework_versions=installed_framework_versions(),
        devices=("cpu", "cuda_optional"),
        limitations=limitations,
    )


def observable_device_memory(device: str) -> tuple[int | None, str]:
    """Return observable bytes and an evidence label, importing torch only on launch."""

    if device == "cpu":
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            return page_size * available_pages, "os_sysconf_available_physical_memory"
        except (AttributeError, OSError, ValueError):
            return None, "not_observable"
    if device == "cuda":
        try:
            import torch  # type: ignore[import-not-found]

            if not torch.cuda.is_available():
                return None, "cuda_unavailable"
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            return int(free_bytes), "torch_cuda_mem_get_info"
        except (ImportError, RuntimeError):
            return None, "not_observable"
    return None, "unsupported_device"
