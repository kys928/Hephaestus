#!/usr/bin/env python3
"""Conservative cleanup worker for the Hephaestus RunPod Network Volume.

Deletes only artifacts that are provably disposable:
- caches outside the governed scientific root;
- Python/test caches anywhere;
- stale temporary/partial files;
- model snapshots explicitly marked cache_persistence=ephemeral.

Scientific results, checkpoints, adapters, manifests, lineage, and the active
adaptation-elasticity run are never deleted by this worker. Potentially large
checkpoint directories are inventoried only and left untouched.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/workspace")
HEPHAESTUS = ROOT / "hephaestus"
SCIENTIFIC = HEPHAESTUS / "scientific" / "v1"
RUN_ID = os.environ.get("HEPHAESTUS_CLEANUP_RUN_ID", "storage-cleanup-unknown")
PROTECTED_RUN_ID = os.environ.get("HEPHAESTUS_PROTECTED_RUN_ID", "").strip()
REPORT_ROOT = SCIENTIFIC / "maintenance" / "storage_cleanup" / RUN_ID
REPORT_PATH = REPORT_ROOT / "report.json"
STALE_SECONDS = 6 * 3600
LOCK_STALE_SECONDS = 24 * 3600

SAFE_CACHE_DIR_NAMES = {
    ".cache",
    "hf_cache",
    "huggingface_cache",
    "pip_cache",
    "torch_extensions",
    "__pycache__",
    ".pytest_cache",
}
ALWAYS_SAFE_DIR_NAMES = {"__pycache__", ".pytest_cache"}
SAFE_TEMP_SUFFIXES = {".partial", ".tmp", ".pyc"}
SAFE_TEMP_NAMES = {".DS_Store"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def disk() -> dict[str, int]:
    usage = shutil.disk_usage(ROOT)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def protected_prefixes() -> list[Path]:
    rows = [REPORT_ROOT]
    if PROTECTED_RUN_ID:
        rows.extend(
            [
                SCIENTIFIC / "adaptation_elasticity" / PROTECTED_RUN_ID,
                SCIENTIFIC / "executions" / PROTECTED_RUN_ID,
            ]
        )
    return rows


def is_protected(path: Path) -> bool:
    resolved = path.resolve()
    return any(is_under(resolved, prefix) for prefix in protected_prefixes())


def apparent_size(path: Path) -> int:
    try:
        if path.is_symlink():
            return 0
        if path.is_file():
            return path.stat().st_size
        proc = subprocess.run(
            ["du", "-x", "-s", "-B1", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return int(proc.stdout.split()[0])
    except Exception:
        pass
    return 0


def du_snapshot() -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["du", "-x", "-B1", "-d", "3", str(HEPHAESTUS)],
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        try:
            size, path = line.split("\t", 1)
            rows.append({"bytes": int(size), "path": path})
        except ValueError:
            continue
    rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
    return rows[:80]


def remove_path(path: Path, reason: str, deleted: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    if is_protected(path):
        return
    size = apparent_size(path)
    try:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            return
        deleted.append({"path": str(path), "reason": reason, "bytes": size})
    except Exception as exc:
        failures.append({"path": str(path), "reason": reason, "error": f"{type(exc).__name__}: {exc}"})


def delete_safe_caches(deleted: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    candidates: list[Path] = []
    for base, dirs, _files in os.walk(ROOT, topdown=True, followlinks=False):
        base_path = Path(base)
        kept: list[str] = []
        for name in dirs:
            path = base_path / name
            if is_protected(path):
                kept.append(name)
                continue
            if name in ALWAYS_SAFE_DIR_NAMES:
                candidates.append(path)
                continue
            if name in SAFE_CACHE_DIR_NAMES and not is_under(path, SCIENTIFIC):
                candidates.append(path)
                continue
            kept.append(name)
        dirs[:] = kept
    for path in sorted(set(candidates), key=lambda p: len(p.parts), reverse=True):
        if path.exists():
            remove_path(path, "disposable_cache", deleted, failures)


def delete_stale_temp_files(deleted: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    cutoff = time.time() - STALE_SECONDS
    lock_cutoff = time.time() - LOCK_STALE_SECONDS
    for base, dirs, files in os.walk(HEPHAESTUS, topdown=True, followlinks=False):
        base_path = Path(base)
        dirs[:] = [name for name in dirs if not is_protected(base_path / name)]
        for name in files:
            path = base_path / name
            if is_protected(path):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            safe = name in SAFE_TEMP_NAMES or path.suffix in SAFE_TEMP_SUFFIXES
            stale_lock = path.suffix == ".lock" and mtime < lock_cutoff
            if (safe and mtime < cutoff) or stale_lock:
                remove_path(path, "stale_temporary_file", deleted, failures)


def delete_explicit_ephemeral_snapshots(deleted: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    for manifest_path in SCIENTIFIC.rglob("snapshot_manifest.json"):
        if is_protected(manifest_path):
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("cache_persistence") != "ephemeral":
            continue
        raw = str(payload.get("snapshot_path") or "")
        if not raw.startswith("/workspace/"):
            continue
        snapshot = Path(raw)
        if snapshot.exists() and is_under(snapshot, ROOT) and not is_protected(snapshot):
            remove_path(snapshot, f"ephemeral_snapshot:{manifest_path}", deleted, failures)


def checkpoint_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs = SCIENTIFIC / "runs"
    if not runs.exists():
        return rows
    for path in runs.rglob("checkpoint*"):
        if path.is_dir():
            rows.append({"path": str(path), "bytes": apparent_size(path), "deleted": False, "reason": "scientific_checkpoint_preserved"})
    rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
    return rows[:100]


def main() -> int:
    try:
        os.nice(10)
    except OSError:
        pass
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    before = disk()
    top_before = du_snapshot()
    deleted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    delete_safe_caches(deleted, failures)
    delete_stale_temp_files(deleted, failures)
    delete_explicit_ephemeral_snapshots(deleted, failures)
    try:
        os.sync()
    except AttributeError:
        pass

    after = disk()
    top_after = du_snapshot()
    checkpoints = checkpoint_inventory()
    deleted_bytes_observed = max(0, after["free_bytes"] - before["free_bytes"])
    report = {
        "report_version": "network-volume-cleanup.v1",
        "status": "completed" if not failures else "completed_with_nonfatal_failures",
        "created_at": now(),
        "cleanup_run_id": RUN_ID,
        "protected_run_id": PROTECTED_RUN_ID or None,
        "policy": {
            "scientific_checkpoints_deleted": False,
            "scientific_adapters_deleted": False,
            "scientific_results_deleted": False,
            "lineage_deleted": False,
            "active_run_deleted": False,
            "safe_categories": ["disposable caches", "Python/test caches", "stale temp/partial files", "explicit ephemeral model snapshots"],
        },
        "before": before,
        "after": after,
        "free_space_gained_bytes": deleted_bytes_observed,
        "deleted_entries": deleted,
        "deleted_entry_count": len(deleted),
        "deleted_entry_apparent_bytes": sum(int(row["bytes"]) for row in deleted),
        "failures": failures,
        "top_usage_before": top_before,
        "top_usage_after": top_after,
        "preserved_checkpoint_inventory": checkpoints,
        "large_reclaim_candidates_not_deleted": checkpoints[:25],
    }
    temporary = REPORT_PATH.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, REPORT_PATH)
    print("NETWORK_VOLUME_CLEANUP_JSON " + json.dumps({
        "status": report["status"],
        "free_space_gained_bytes": deleted_bytes_observed,
        "deleted_entry_count": len(deleted),
        "failure_count": len(failures),
        "report_path": str(REPORT_PATH),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
