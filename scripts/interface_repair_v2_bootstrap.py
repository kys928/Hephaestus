#!/usr/bin/env python3
"""Persistent bootstrap helpers for Interface Repair V2.

Execution-only optimization: reuse the immutable Hugging Face snapshot and Phase-I
adapter directly from the Network Volume instead of rematerializing them into a
per-run /opt tree. Scientific inputs, revisions and adapter digests stay frozen.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verified_archive(path: Path, *, expected_bytes: int, expected_sha256: str) -> bool:
    return path.is_file() and path.stat().st_size == expected_bytes and _sha256(path) == expected_sha256


def load_stack_compatible(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Load the certified Phase-I stack using its actual registry schema.

    Historical V1 execution expected a now-obsolete top-level ``source_run_id``.
    The certified registry stores the same immutable identifier at
    ``source_experiment.run_id``. Normalize only in memory so V2 can consume the
    certified registry without mutating it or changing any scientific input.
    """
    path = ROOT / str(cfg["source_phase_i_stack"]["registry_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    source = data.get("source_experiment")
    observed = data.get("source_run_id")
    if not observed and isinstance(source, dict):
        observed = source.get("run_id")
    expected = cfg["source_phase_i_stack"]["required_source_run_id"]
    if observed != expected:
        raise RuntimeError(f"source Phase-I run mismatch: {observed!r} != {expected!r}")
    if cfg["source_phase_i_stack"].get("production_certified_required") and not data.get("production_certified"):
        raise RuntimeError("source Phase-I stack is not production certified")
    if data.get("automatic_role_dispatch_enabled"):
        raise RuntimeError("interface repair requires automatic role dispatch to remain disabled")
    normalized = dict(data)
    normalized["source_run_id"] = str(observed)
    return normalized


def materialize_parent_cached(client: Any, role_spec: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    """Return immutable base-model and adapter directories from persistent caches.

    ``root`` is retained in the signature for drop-in compatibility with V1 but is
    intentionally not used for the base model. The HF snapshot cache lives on the
    mounted Network Volume through HF_HOME/HUGGINGFACE_HUB_CACHE.
    """
    del root
    from huggingface_hub import snapshot_download
    import run_interface_repair_v1 as v1

    model_id = str(role_spec["model_id"])
    revision = str(role_spec["revision"])

    try:
        snapshot = snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
    except Exception:
        snapshot = snapshot_download(repo_id=model_id, revision=revision)
    base = Path(snapshot)
    if not (base / "config.json").is_file():
        raise RuntimeError(f"persistent HF snapshot is incomplete for {model_id}@{revision}")

    adapter = role_spec["adapter"]
    expected_bytes = int(adapter["bytes"])
    expected_sha = str(adapter["sha256"])
    cache_root = Path(os.environ.get(
        "HEPHAESTUS_V2_ADAPTER_CACHE",
        "/workspace/hephaestus-cache/interface-repair-v2/adapters",
    ))
    entry = cache_root / expected_sha
    entry.mkdir(parents=True, exist_ok=True)
    archive = entry / "parent-adapter.tar.gz"

    if not _verified_archive(archive, expected_bytes=expected_bytes, expected_sha256=expected_sha):
        tmp = entry / f"parent-adapter.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            client.download_file(v1.bucket(), str(adapter["s3_key"]), str(tmp))
            if not _verified_archive(tmp, expected_bytes=expected_bytes, expected_sha256=expected_sha):
                raise RuntimeError("downloaded parent adapter failed frozen byte/SHA verification")
            os.replace(tmp, archive)
        finally:
            tmp.unlink(missing_ok=True)

    extracted = entry / "extracted"
    adapter_dir = extracted / "adapter"
    ready = entry / ".adapter-ready"
    if not (ready.is_file() and adapter_dir.is_dir()):
        staging = entry / f"extract.{os.getpid()}.{time.time_ns()}"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            observed = v1.safe_extract(archive, staging)
            if observed.name != "adapter" or observed.parent != staging:
                normalized = staging / "adapter"
                if normalized.exists():
                    shutil.rmtree(normalized)
                shutil.move(str(observed), str(normalized))
            if extracted.exists():
                shutil.rmtree(extracted)
            os.replace(staging, extracted)
            ready.write_text(expected_sha + "\n", encoding="utf-8")
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    if not adapter_dir.is_dir():
        raise RuntimeError("persistent parent adapter extraction is incomplete")
    return base, adapter_dir
