#!/usr/bin/env python3
"""Cold-custody copy of an original complete CPU campaign onto an operator-controlled mount.

A moved cold copy is NOT a resumable workflow: checkpoint paths and run bindings are
intentionally not rewritten. Choose a genuinely persistent mount at initial launch.
No S3/API calls, no hidden network transfer, no overwrite of existing evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _members(root):
    result = {}
    for p in root.rglob("*"):
        if p.is_symlink():
            raise PermissionError("evidence archive must not include symlinks")
        if p.is_file():
            result[p.relative_to(root).as_posix()] = p
    return result


def verify_original_evidence(root):
    root = Path(root).resolve(strict=True)
    report_path, manifest_path = root/"cpu_campaign_report.json", root/"archive_sha256.json"
    if not report_path.is_file() or not manifest_path.is_file() or not (root/"state.sqlite").is_file():
        raise FileNotFoundError("completed CPU report, archive manifest and native SQLite required")
    report = json.loads(report_path.read_bytes())
    evidence = report.get("evidence") or {}
    if (report.get("status") != "completed" or report.get("completed_operations") != 45 or
            evidence.get("verified_operations") != 45 or evidence.get("verified_domain_operations") != 40 or
            evidence.get("claim_scope") != "internal_prospective_generated_task_research" or
            evidence.get("independent_scientific_certification") is not False or
            evidence.get("five_frozen_cuda_roles_executed") is not False):
        raise PermissionError("only authentic complete internal CPU research records can be cold-archived")
    expected = json.loads(manifest_path.read_bytes())
    observed = _members(root)
    observed.pop("archive_sha256.json", None)
    if not isinstance(expected, dict) or set(expected) != set(observed):
        raise PermissionError("archived member inventory differs from original evidence")
    for relative, path in observed.items():
        if expected[relative] != _digest(path):
            raise PermissionError("archive member SHA256 mismatch: " + relative)
    # Read-only integrity check, never a fabricated trace replay.
    uri = (root/"state.sqlite").as_uri() + "?mode=ro&immutable=1"
    db = sqlite3.connect(uri, uri=True)
    try:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("original campaign SQLite quick_check failed")
    finally:
        db.close()
    return {"schema": "cpu-cold-custody-verification.v1",
            "archive_manifest_sha256": _digest(manifest_path), "member_count": len(observed),
            "workflow_id": report["workflow_id"], "original_operations": 45,
            "externally_certified": False, "resumable_from_copied_path": False}


def retain(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise ValueError("new evidence destination must be unused absolute path")
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ValueError("operator-controlled persistent parent must exist and not be a symlink")
    if destination.parent.resolve() == source or source in destination.parents:
        raise ValueError("cannot write archive inside source workspace")
    original = verify_original_evidence(source)
    partial = destination.with_name(destination.name + ".uncommitted")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError("uncommitted custody copy needs inspection")
    partial.mkdir(mode=0o700)
    try:
        for relative, p in _members(source).items():
            target = partial/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            os.chmod(target, 0o600)
        measured = verify_original_evidence(partial)
        if measured != original or verify_original_evidence(source) != original:
            raise PermissionError("evidence source changed during cold-custody transfer")
        partial.rename(destination)
    except BaseException:
        # Never silently erase partial custody evidence after an I/O incident.
        raise
    return original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(retain(args.source, args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
