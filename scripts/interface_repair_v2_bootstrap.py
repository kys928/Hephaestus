#!/usr/bin/env python3
"""Bootstrap helpers for Interface Repair V2.

Execution-only optimization: reuse immutable model/adapter artifacts from the
Network Volume when they are already complete, but never require the persistent
volume to have enough free space for a new model download. Missing immutable
snapshots fall back to the pod's ephemeral container disk. Scientific inputs,
revisions and adapter digests stay frozen.
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


def _adapter_from_entry(client: Any, role_spec: Mapping[str, Any], entry: Path) -> Path:
    import run_interface_repair_v1 as v1

    adapter = role_spec["adapter"]
    expected_bytes = int(adapter["bytes"])
    expected_sha = str(adapter["sha256"])
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
        raise RuntimeError("parent adapter extraction is incomplete")
    return adapter_dir


def materialize_parent_cached(client: Any, role_spec: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    """Resolve frozen model + adapter without requiring persistent free space.

    Read a complete immutable model snapshot from the persistent HF cache when
    available. If that exact revision is absent/incomplete, download it to the
    pod's 180-GB ephemeral container disk instead of expanding the quota-bound
    Network Volume. The same strategy is used for the parent adapter: persistent
    cache is read when already complete; otherwise a verified ephemeral copy is
    used. No scientific artifact or evidence is deleted or mutated.
    """
    from huggingface_hub import snapshot_download

    model_id = str(role_spec["model_id"])
    revision = str(role_spec["revision"])

    cache_source = "persistent"
    try:
        snapshot = snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
    except Exception:
        cache_source = "ephemeral"
        ephemeral_hf = Path(os.environ.get("HEPHAESTUS_V2_EPHEMERAL_HF_CACHE", "/opt/hephaestus-hf-cache"))
        ephemeral_hf.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_download(repo_id=model_id, revision=revision, cache_dir=str(ephemeral_hf))
    base = Path(snapshot)
    if not (base / "config.json").is_file():
        raise RuntimeError(f"HF snapshot is incomplete for {model_id}@{revision}")
    print(json.dumps({
        "event": "interface_repair_v2_model_cache",
        "model_id": model_id,
        "revision": revision,
        "source": cache_source,
        "path": str(base),
    }, sort_keys=True), flush=True)

    adapter = role_spec["adapter"]
    expected_sha = str(adapter["sha256"])
    persistent_root = Path(os.environ.get(
        "HEPHAESTUS_V2_ADAPTER_CACHE",
        "/workspace/hephaestus-cache/interface-repair-v2/adapters",
    ))
    persistent_entry = persistent_root / expected_sha
    persistent_adapter = persistent_entry / "extracted" / "adapter"
    persistent_ready = persistent_entry / ".adapter-ready"
    archive = persistent_entry / "parent-adapter.tar.gz"
    expected_bytes = int(adapter["bytes"])

    if (
        persistent_ready.is_file()
        and persistent_adapter.is_dir()
        and _verified_archive(archive, expected_bytes=expected_bytes, expected_sha256=expected_sha)
    ):
        adapter_dir = persistent_adapter
        adapter_source = "persistent"
    else:
        adapter_source = "ephemeral"
        ephemeral_entry = root / "adapter-cache" / expected_sha
        adapter_dir = _adapter_from_entry(client, role_spec, ephemeral_entry)

    print(json.dumps({
        "event": "interface_repair_v2_adapter_cache",
        "sha256": expected_sha,
        "source": adapter_source,
        "path": str(adapter_dir),
    }, sort_keys=True), flush=True)
    return base, adapter_dir
