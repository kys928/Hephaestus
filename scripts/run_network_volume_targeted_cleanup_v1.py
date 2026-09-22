#!/usr/bin/env python3
"""Delete only the two S3-reviewed reconstructable model-cache trees.

This worker runs on a Pod with the persistent Network Volume mounted at
/workspace.  It has a hard-coded allowlist in addition to the checked-in
manifest so a later manifest edit cannot broaden deletion authority by itself.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/workspace")
SCIENTIFIC = ROOT / "hephaestus" / "scientific" / "v1"
MANIFEST_PATH = Path("configs/maintenance/network_volume_targeted_cleanup_v1.json")
RUN_ID = os.environ.get("HEPHAESTUS_TARGETED_CLEANUP_RUN_ID", "targeted-cleanup-unknown").strip()
REPORT_ROOT = SCIENTIFIC / "maintenance" / "targeted_storage_cleanup" / RUN_ID
REPORT_PATH = REPORT_ROOT / "report.json"

HARD_ALLOWLIST = {
    "/workspace/hephaestus/scientific/v1/model_cache/huggingface",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/hf_cache",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _disk() -> dict[str, int]:
    usage = shutil.disk_usage(ROOT)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _du(path: Path) -> int:
    if not path.exists():
        return 0
    proc = subprocess.run(
        ["du", "-x", "-s", "-B1", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"du failed for {path}: {proc.stderr.strip()}")
    return int(proc.stdout.split()[0])


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _file_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for _base, _dirs, files in os.walk(path, followlinks=False):
        count += len(files)
    return count


def _load_manifest() -> tuple[dict[str, Any], str]:
    raw = MANIFEST_PATH.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("cleanup_id") != "network_volume_targeted_cleanup_v1":
        raise RuntimeError("unexpected targeted cleanup manifest id")
    if manifest.get("volume_id") != "cviwpryzao" or manifest.get("datacenter_id") != "EU-CZ-1":
        raise RuntimeError("targeted cleanup manifest points at the wrong volume/datacenter")
    rows = manifest.get("delete_exact_prefixes")
    if not isinstance(rows, list):
        raise RuntimeError("targeted cleanup manifest lacks delete_exact_prefixes")
    observed = {str(row.get("path")) for row in rows if isinstance(row, dict)}
    if observed != HARD_ALLOWLIST:
        raise RuntimeError(f"manifest deletion set differs from hard allowlist: {sorted(observed)}")
    if manifest.get("safety", {}).get("delete_only_exact_allowlist") is not True:
        raise RuntimeError("manifest does not require exact allowlist deletion")
    return manifest, _sha(raw)


def main() -> int:
    manifest, manifest_sha = _load_manifest()
    active_id = str(manifest["protected_active_run_id"])
    active_root = SCIENTIFIC / "adaptation_elasticity" / active_id
    active_execution = SCIENTIFIC / "executions" / active_id
    if not active_root.exists():
        raise RuntimeError(f"protected active elasticity path is missing: {active_root}")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    before = _disk()
    active_before = {
        "path": str(active_root),
        "exists": active_root.exists(),
        "files": _file_count(active_root),
        "apparent_bytes": _du(active_root),
        "execution_path_exists": active_execution.exists(),
    }

    targets: list[dict[str, Any]] = []
    for row in manifest["delete_exact_prefixes"]:
        target = Path(str(row["path"]))
        canonical = str(target)
        if canonical not in HARD_ALLOWLIST:
            raise RuntimeError(f"target is not hard allowlisted: {target}")
        if not target.is_absolute() or not _is_under(target, SCIENTIFIC):
            raise RuntimeError(f"target escapes governed scientific root: {target}")
        if target.is_symlink():
            raise RuntimeError(f"target root is a symlink: {target}")
        # Never permit an allowed target to contain either protected active tree.
        if _is_under(active_root, target) or _is_under(active_execution, target):
            raise RuntimeError(f"target would contain protected active state: {target}")
        expected = int(row["s3_observed_bytes"])
        present = target.exists()
        measured = _du(target) if present else 0
        targets.append(
            {
                "path": canonical,
                "s3_prefix": row["s3_prefix"],
                "reason": row["reason"],
                "s3_observed_bytes": expected,
                "present_before": present,
                "apparent_bytes_before": measured,
                "file_count_before": _file_count(target) if present else 0,
            }
        )

    deleted: list[dict[str, Any]] = []
    for target_row in targets:
        target = Path(target_row["path"])
        if target.exists():
            shutil.rmtree(target)
            if target.exists():
                raise RuntimeError(f"target still exists after rmtree: {target}")
            deleted.append(dict(target_row))

    try:
        os.sync()
    except AttributeError:
        pass

    if not active_root.exists():
        raise RuntimeError("protected elasticity run disappeared during targeted cleanup")
    after = _disk()
    active_after = {
        "path": str(active_root),
        "exists": active_root.exists(),
        "files": _file_count(active_root),
        "apparent_bytes": _du(active_root),
        "execution_path_exists": active_execution.exists(),
    }
    report = {
        "report_version": "network-volume-targeted-cleanup.v1",
        "status": "completed",
        "created_at": _now(),
        "cleanup_run_id": RUN_ID,
        "manifest_sha256": manifest_sha,
        "manifest_basis": manifest["basis"],
        "protected_active_run_id": active_id,
        "hard_allowlist": sorted(HARD_ALLOWLIST),
        "before": before,
        "after": after,
        "free_space_gained_bytes": max(0, after["free_bytes"] - before["free_bytes"]),
        "targets": targets,
        "deleted": deleted,
        "deleted_target_count": len(deleted),
        "deleted_apparent_bytes": sum(int(row["apparent_bytes_before"]) for row in deleted),
        "s3_reviewed_bytes": sum(int(row["s3_observed_bytes"]) for row in targets),
        "active_before": active_before,
        "active_after": active_after,
        "safety": manifest["safety"],
    }
    temporary = REPORT_PATH.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, REPORT_PATH)
    print(
        "NETWORK_VOLUME_TARGETED_CLEANUP_JSON "
        + json.dumps(
            {
                "status": "completed",
                "deleted_target_count": len(deleted),
                "free_space_gained_bytes": report["free_space_gained_bytes"],
                "deleted_apparent_bytes": report["deleted_apparent_bytes"],
                "report_path": str(REPORT_PATH),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
