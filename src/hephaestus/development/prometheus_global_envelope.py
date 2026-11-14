"""Run inside ONE independently supervised, pinned Global-volume CUDA Pod.

Runs the original decision-first Prometheus worker with authenticated historical
source bytes. Copies every available raw inference receipt and failure artifact
to private regional S3 with SHA-256 readback BEFORE writing a final manifest.
No training/promotion/protected holdout work is permitted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

from .prometheus_global_supervision import (
    GLOBAL_ID, S3_BUCKET, SOURCE_D, SOURCE_PARENT, authenticate_sources,
    campaign_prefix, canonical, get_bytes, iso_now, put_once, s3_client, sha,
)

LOCAL = Path("/tmp/prometheus-global-campaign")
GLOBAL = Path("/workspace-global")


def _read_input(s3, prefix: str, name: str, expected: str) -> bytes:
    raw = get_bytes(s3, prefix + "/source/" + name, maximum=40_000)
    if sha(raw) != expected:
        raise PermissionError("original source report mismatch on GPU worker")
    return raw


def _persist_tree(s3, prefix: str, root: Path, *, result_code: int,
                  elapsed_seconds: float, metadata: dict) -> dict:
    files = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PermissionError("worker evidence symlink forbidden")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if not re.fullmatch(r"[A-Za-z0-9._/-]{1,240}", rel) or ".." in Path(rel).parts:
            raise PermissionError("unadmitted worker evidence path")
        size = path.stat().st_size
        if size > 30 * 1024 * 1024:
            raise PermissionError("oversized individual worker evidence object")
        total += size
        if total > 160 * 1024 * 1024 or len(files) >= 1000:
            raise PermissionError("bounded evidence publication exceeded")
        body = path.read_bytes()
        digest = put_once(s3, prefix + "/result/files/" + rel, body)
        files.append({"path": rel, "bytes": len(body), "sha256": digest})
    if not files:
        raise PermissionError("empty scientific evidence bundle cannot be admitted")
    manifest = {
        "schema": "prometheus-global-evidence-manifest.v1",
        "prefix": prefix, "global_volume_id": GLOBAL_ID,
        "regional_custody_bucket": S3_BUCKET,
        "source_parent_sha256": SOURCE_PARENT,
        "source_D_sha256": SOURCE_D,
        "worker_exit_code": result_code, "elapsed_seconds": elapsed_seconds,
        "model_autonomy_proven": False,
        "candidate_repair_claimed": False,
        "independent_certification": False,
        "metadata": metadata, "files": files, "sealed_at": iso_now(),
    }
    put_once(s3, prefix + "/result/manifest.json", canonical(manifest))
    return manifest


def run() -> int:
    campaign = os.environ["PROMETHEUS_GLOBAL_CAMPAIGN"]
    source_sha = os.environ["PROMETHEUS_SOURCE_SHA"]
    prefix = campaign_prefix(campaign, source_sha)
    duration = int(os.environ["PROMETHEUS_WORKER_SECONDS"])
    if not 600 <= duration <= 5100:
        raise PermissionError("worker runtime outside bounded lease")
    if os.environ.get("RUNPOD_GLOBAL_VOLUME_ID") != GLOBAL_ID or not GLOBAL.is_mount():
        raise PermissionError("Global Volume absent or wrong external ID")
    for p in (LOCAL, LOCAL / "source", LOCAL / "output", LOCAL / "logs"):
        p.mkdir(parents=True, exist_ok=True)
    if LOCAL.resolve().is_relative_to(GLOBAL.resolve()):
        raise PermissionError("worker writes may not use object-backed Global")
    s3 = s3_client()
    started = time.monotonic()
    status = {"stage": "worker_preflight", "source_sha": source_sha}
    try:
        # Exact source bytes copied from unmodified GitHub Actions artifacts
        # by the off-pod controller, then reauthenticated inside the GPU Pod.
        parent = _read_input(s3, prefix, "original-parent-dev.json", SOURCE_PARENT)
        historical_d = _read_input(s3, prefix, "historical-d-dev.json", SOURCE_D)
        authenticate_sources(parent, historical_d)
        (LOCAL / "source/original-parent-dev.json").write_bytes(parent)
        (LOCAL / "source/historical-d-dev.json").write_bytes(historical_d)
        (LOCAL / "source/source-custody.json").write_bytes(canonical({
            "schema": "prometheus-global-worker-inputs.v1",
            "parent": SOURCE_PARENT, "historical_D": SOURCE_D, "source_commit": source_sha,
            "historical_exposed_not_fresh_holdout": True,
        }))
        status["stage"] = "authenticated_cuda_science"
        args = [
            sys.executable, "-u", "-m", "hephaestus.development.prometheus_gpu_campaign",
            "--parent-report", str(LOCAL / "source/original-parent-dev.json"),
            "--d-report", str(LOCAL / "source/historical-d-dev.json"),
            "--out", str(LOCAL / "output/cycle"),
            "--global-volume-root", str(GLOBAL),
            "--global-volume-id", GLOBAL_ID,
            "--adapter-cache", str(LOCAL / "cache/adapters"),
            "--hf-cache", "/tmp/prometheus-hf-cache",
        ]
        with (LOCAL / "logs/scientific-worker.log").open("wb") as trace:
            try:
                completed = subprocess.run(args, stdout=trace, stderr=subprocess.STDOUT,
                                           timeout=duration, check=False)
                code = int(completed.returncode)
            except subprocess.TimeoutExpired:
                code = 124
                status["error_class"] = "ScientificWorkerTimeout"
        status["stage"] = "science_finished"
    except Exception as exc:
        code = 1
        status["error_class"] = type(exc).__name__
        status["stage"] = "failed_preflight_or_execution"
    # Never log traceback, secrets, user data, or raw model text to public Actions.
    status.update({"worker_exit_code": code, "source_parent_sha256": SOURCE_PARENT,
                   "source_D_sha256": SOURCE_D, "global_volume_id": GLOBAL_ID,
                   "elapsed_seconds": time.monotonic() - started})
    (LOCAL / "logs/worker-status.json").write_bytes(canonical(status))
    # Build a single bounded retained tree with source provenance and full raw
    # reasoning/inference receipts, including the original failed run artifacts.
    retained = LOCAL / "retained"
    retained.mkdir(exist_ok=True)
    import shutil
    for label in ("source", "output", "logs"):
        original = LOCAL / label
        if original.exists():
            shutil.copytree(original, retained / label, symlinks=False, dirs_exist_ok=True)
    manifest = _persist_tree(s3, prefix, retained, result_code=code,
                             elapsed_seconds=time.monotonic() - started, metadata=status)
    put_once(s3, prefix + "/status/worker-exit.json", canonical({
        "schema": "prometheus-global-worker-exit.v1",
        "prefix": prefix, "code": code, "manifest_sha256": sha(canonical(manifest)),
        "finished_at": iso_now(),
    }))
    print("PROMETHEUS_GLOBAL_EVIDENCE_PERSISTED=true", flush=True)
    print("PROMETHEUS_GLOBAL_WORKER_EXIT=" + str(code), flush=True)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        # If S3 itself is unavailable the watchdog still enforces termination.
        print("PROMETHEUS_GLOBAL_WORKER_INCOMPLETE=" + type(exc).__name__, flush=True)
        raise SystemExit(1)
