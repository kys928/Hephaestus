#!/usr/bin/env python3
"""HEPHAESTUS HF -> RUNPOD GLOBAL VOLUME, v3 (no command-line arguments).

One-command, resumable, independently SHA-256-verified restore of the 8,986
artifacts listed in the private HF EXPORT_MANIFEST.json.

Run ONLY on a new Pod mounting global volume cmw51pjsk000007l9a0mv1jga.
    python3 -u /tmp/hephaestus_hf_to_global_v3.py

Downloads in SMALL bounded batches to ephemeral /tmp, verifies local SHA-256,
IMMEDIATELY writes each verified file to a content-namespaced Global Volume
directory and re-reads SHA-256. Completed files survive Pod termination;
interrupted downloads do not need to be restarted from scratch.
Does not modify source Czech volume, existing r9 archive, HF repositories,
existing Hephaestus state, or unrelated Pods. No git clone, ZIP/TAR, API keys
in arguments, or remote deletion. Private read token is not saved.

The generated RESTORE_RECEIPT.json is an integrity report, NOT a scientific
model certificate, a live-worker test, or proof all Czech files were exported.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

EXPECTED_USER = "DavidDulovic"
EXPECTED_SOURCE_VOLUME = "cviwpryzao"
GLOBAL_VOLUME_ID = "cmw51pjsk000007l9a0mv1jga"
# Observed on the verified Global Volume mount; acts as an additional identity gate.
GLOBAL_MOUNT_SOURCE = "b5eb0ee5-bc81-4a25-bc3b-08d0c0607649"
RESTORE_SUBPATH = Path("hephaestus/hf-restores")
KNOWN_GLOBAL_ARCHIVE = Path("hephaestus/artifacts/archives/original_cuda_45/") \
    / "cuda-json-45-original-five-20261030-r9.tar"
HF_REPOS = {
    "models": (f"{EXPECTED_USER}/hephaestus-trained-models", "model", 869),
    "datasets": (f"{EXPECTED_USER}/hephaestus-training-data", "dataset", 4),
    "evidence": (f"{EXPECTED_USER}/hephaestus-scientific-evidence", "dataset", 8113),
}
TOTAL_FILES = sum(v[2] for v in HF_REPOS.values())
MAX_TOTAL_BYTES = 30 * 1024**3
TMP_ROOT = Path("/tmp/hephaestus-hf-global-restore-v3")
DOWNLOAD_WORKERS = 6
DOWNLOAD_WINDOW = 18
BUF = 4 * 1024 * 1024
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def say(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(BUF):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def safe_remote_path(value: object) -> Path:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError("Invalid empty/long remote path")
    if "\\" in value or "\x00" in value or value.startswith("/") or "//" in value:
        raise ValueError(f"Unsafe remote path: {value!r}")
    parts = PurePosixPath(value).parts
    if not parts or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError(f"Unsafe remote path: {value!r}")
    # The source exporter excluded dotfiles themselves, but can include
    # ordinary files inside dot-prefixed directories. Such names are valid.
    # Exact "." and ".." components are blocked above, so hidden directory
    # names (for example ".reports") cannot escape the restore root.
    if parts[0] not in {"scientific", "research", "legacy"}:
        raise ValueError("Unexpected source prefix in HF manifest")
    return Path(*parts)


def parse_manifest(raw: bytes, *, strict_counts: bool = True) -> dict[str, list[dict]]:
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("Oversized EXPORT_MANIFEST.json")
    def no_duplicate_pairs(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("Duplicate JSON key in manifest")
            result[k] = v
        return result
    obj = json.loads(raw, object_pairs_hook=no_duplicate_pairs)
    if not isinstance(obj, dict) or obj.get("schema") != "hephaestus-hf-export-v1":
        raise ValueError("Not the expected Hephaestus HF export manifest")
    if obj.get("source_volume_id") != EXPECTED_SOURCE_VOLUME:
        raise ValueError("Wrong source-volume ID in export manifest")
    entries = obj.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError("Manifest has no artifact list")
    result: dict[str, list[dict]] = {group: [] for group in HF_REPOS}
    seen = set()
    total_size = 0
    by_repo = {(rid, rtype): group for group, (rid, rtype, _) in HF_REPOS.items()}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Malformed manifest artifact")
        key = (entry.get("repo"), entry.get("repo_type"))
        group = by_repo.get(key)
        if group is None:
            raise ValueError("Unexpected HF repository or repository type")
        remote = safe_remote_path(entry.get("path"))
        size, digest = entry.get("bytes"), entry.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise ValueError("Invalid artifact byte count or SHA-256")
        if entry.get("remote_verification") != "verified":
            raise ValueError("Export includes a file lacking strong remote metadata verification")
        identity = (group, remote.as_posix())
        if identity in seen:
            raise ValueError("Duplicate artifact path in manifest")
        seen.add(identity)
        total_size += size
        if total_size > MAX_TOTAL_BYTES:
            raise ValueError("Export manifest exceeds the 30 GiB safety bound")
        result[group].append({"path": remote, "bytes": size, "sha256": digest})
    if strict_counts:
        if len(entries) != TOTAL_FILES or any(len(result[g]) != HF_REPOS[g][2] for g in HF_REPOS):
            raise ValueError("Expected 869 models, 4 datasets, 8113 evidence files; manifest differs")
    for group in result:
        result[group].sort(key=lambda row: row["path"].as_posix())
    return result


def find_global_volume() -> Path:
    candidates = (Path("/workspace"), Path("/workspace-global"))
    accepted = []
    for root in candidates:
        if not root.is_dir() or root.is_symlink():
            continue
        try:
            raw = subprocess.check_output(
                ["findmnt", "-T", str(root), "-n", "-o", "SOURCE,FSTYPE,TARGET"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip().split()
        except (OSError, subprocess.CalledProcessError):
            continue
        if len(raw) < 3:
            continue
        source, fstype, mounted_at = raw[0], raw[1], raw[2]
        # A Pod ephemeral overlay, regional FUSE and unrelated Global Volume
        # must NEVER be silently accepted as the destination.
        if (GLOBAL_MOUNT_SOURCE in source and "geesefs" in fstype
                and mounted_at == str(root)
                and (root / KNOWN_GLOBAL_ARCHIVE).is_file()):
            accepted.append(root)
    if len(accepted) != 1:
        raise RuntimeError(
            "Expected Global Volume was not uniquely confirmed at /workspace "
            "or /workspace-global. Verify the Pod mounts volume " + GLOBAL_VOLUME_ID
            + " and the earlier verified r9 archive is visible; refusing to write."
        )
    env_id = os.environ.get("RUNPOD_GLOBAL_VOLUME_ID", "")
    if env_id and env_id != GLOBAL_VOLUME_ID:
        raise RuntimeError("Pod environment advertises a DIFFERENT Global Volume ID")
    return accepted[0]


def verified_path(path: Path, meta: dict) -> bool:
    try:
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_size != meta["bytes"]:
            return False
        digest, byte_size = sha256_file(path)
        return digest == meta["sha256"] and byte_size == meta["bytes"]
    except FileNotFoundError:
        return False


def ensure_safe_parents(base: Path, target: Path) -> None:
    if not target.is_relative_to(base) or target == base:
        raise ValueError("Target escaped dedicated restore directory")
    relative = target.relative_to(base)
    current = base
    if current.is_symlink():
        raise RuntimeError("Restore root is a symlink")
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise RuntimeError("Refusing path through a symlink")
        if current.exists() and not current.is_dir():
            raise RuntimeError("Restore directory path is occupied by a non-directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise RuntimeError("Refusing symlink restore target")


def restore_file(stage: Path, dest: Path, meta: dict, base: Path) -> None:
    if not verified_path(stage, meta):
        raise RuntimeError(f"STAGED BYTES INVALID: {meta['path']}")
    ensure_safe_parents(base, dest)
    if dest.exists():
        if not verified_path(dest, meta):
            raise RuntimeError(f"EXISTING DESTINATION CONFLICT: {dest}; will not overwrite")
        return
    # Single-writer protocol. Global Volumes do not support atomic renames or
    # distributed locking. If interrupted midway, an incomplete file may be
    # visible; no restore receipt is published until *all* hashes match.
    with stage.open("rb") as source, dest.open("xb") as output:
        while block := source.read(BUF):
            output.write(block)
        output.flush()
    for attempt in range(3):
        if verified_path(dest, meta):
            return
        if attempt < 2:
            time.sleep(3)
    raise RuntimeError(f"GLOBAL VOLUME READ-BACK FAILED: {dest}; stop, do not delete source")


def install_client():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        say("Installing Hugging Face client locally on the Pod (without pip download cache)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--no-cache-dir", "huggingface_hub[hf_xet]>=0.36,<3",
        ])
    from huggingface_hub import HfApi, hf_hub_download
    return HfApi, hf_hub_download


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_XET_CACHE", str(TMP_ROOT / "xet"))
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    os.environ.setdefault("HF_XET_SHARD_CACHE_SIZE_LIMIT", str(512 * 1024 * 1024))
    global_root = find_global_volume()
    say(f"Verified RunPod GLOBAL volume {GLOBAL_VOLUME_ID} mounted at {global_root}")
    say("Read-only on Hugging Face. No source CZ volume or existing Global Volume archive is modified.")
    HfApi, hf_hub_download = install_client()
    token = getpass.getpass("Paste a Hugging Face READ token (hidden, not saved): ").strip()
    if not token:
        raise RuntimeError("Missing token")
    api = HfApi(token=token)
    user = api.whoami().get("name")
    if user != EXPECTED_USER:
        raise PermissionError("Wrong Hugging Face account; refusing private restore")
    say(f"Authenticated to private Hugging Face repositories as {EXPECTED_USER}.")
    revisions = {}
    for group, (repo_id, repo_type, _) in HF_REPOS.items():
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
        if info.private is not True or not info.sha or not COMMIT_RE.fullmatch(info.sha):
            raise RuntimeError(f"Repo isn't confirmed PRIVATE or its commit SHA is missing: {repo_id}")
        revisions[group] = info.sha
        say(f"  {group:8s} {repo_id} @{info.sha[:12]}")
    manifest_file = Path(hf_hub_download(
        repo_id=HF_REPOS["evidence"][0], repo_type="dataset",
        filename="EXPORT_MANIFEST.json", revision=revisions["evidence"],
        token=token, cache_dir=str(TMP_ROOT / "manifest-cache"),
    ))
    raw = manifest_file.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    records = parse_manifest(raw)
    dest_base = global_root / RESTORE_SUBPATH / ("export-" + manifest_sha[:24])
    say("\n=== PINNED PRIVATE RESTORE PLAN ===")
    total = 0
    for group, rows in records.items():
        size = sum(row["bytes"] for row in rows)
        total += size
        say(f"  {group:8s} {len(rows):5d} files, {size / 1024**3:7.2f} GiB")
    say(f"  TOTAL     {TOTAL_FILES:5d} files, {total / 1024**3:7.2f} GiB")
    say(f"  Manifest SHA-256: {manifest_sha}")
    say(f"  Global destination: {dest_base}")
    say("  Existing files at other Global Volume paths will not be overwritten.")
    consent = input("\nType RESTORE to download and verify these files: ").strip()
    if consent != "RESTORE":
        say("Cancelled. No Global Volume files were written.")
        return 1
    if dest_base.is_symlink():
        raise RuntimeError("Destination restore directory is a symlink")
    dest_base.mkdir(parents=True, exist_ok=True)
    for group, rows in records.items():
        group_dir = dest_base / group
        if group_dir.is_symlink():
            raise RuntimeError("Unsafe group directory")
        group_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for row in rows:
            dest = group_dir / row["path"]
            if dest.exists() or dest.is_symlink():
                if not verified_path(dest, row):
                    raise RuntimeError(f"Existing restore file conflicts; refusing overwrite: {dest}")
            else:
                missing.append(row)
        say(f"\n{group}: {len(rows)-len(missing)}/{len(rows)} verified on Global Volume; {len(missing)} to download.")
        if not missing:
            continue
        # Unlike v2, never stage the entire 8,113-file evidence collection.
        # Each small window is copied and SHA-256 read-back-verified on Global
        # Volume before the next window starts. A terminated Pod loses at most
        # the current window's transient downloads.
        local_stage = TMP_ROOT / "downloads" / group
        local_stage.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(local_stage).free
        max_file = max(row["bytes"] for row in missing)
        # A bounded number of large Xet downloads plus safety headroom.
        required = max(3 * 1024**3, max_file * DOWNLOAD_WINDOW + 1024**3)
        if free < required:
            raise RuntimeError(
                f"Not enough ephemeral /tmp space for bounded restore: need "
                f"{required/1024**3:.1f} GiB free, have {free/1024**3:.1f}. "
                "Increase CONTAINER DISK; Global Volume is unaffected."
            )
        repo_id, repo_type, _ = HF_REPOS[group]
        say(f"Downloading {group} in windows of {DOWNLOAD_WINDOW} files ("
            f"{DOWNLOAD_WORKERS} workers); saving each verified file to Global Volume immediately...")
        completed = 0
        previous_verified = len(rows) - len(missing)

        def download_one(row: dict) -> Path:
            path = Path(hf_hub_download(
                repo_id=repo_id, repo_type=repo_type, revision=revisions[group],
                token=token, filename=row["path"].as_posix(),
                local_dir=str(local_stage),
            ))
            expected = local_stage / row["path"]
            if path.resolve() != expected.resolve() or not path.is_file() or path.is_symlink():
                raise RuntimeError("Unexpected Hugging Face local download path")
            return path

        for offset in range(0, len(missing), DOWNLOAD_WINDOW):
            window = missing[offset:offset + DOWNLOAD_WINDOW]
            with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
                futures = {pool.submit(download_one, row): row for row in window}
                for future in as_completed(futures):
                    row = futures[future]
                    src = future.result()
                    dest = group_dir / row["path"]
                    restore_file(src, dest, row, dest_base)
                    # Only local ephemeral stage is removed; persistent Global
                    # Volume copy has already passed independent SHA-256.
                    src.unlink()
                    completed += 1
                    if completed % 50 == 0 or completed == len(missing):
                        say(f"  {group}: {completed}/{len(missing)} newly verified "
                            f"(total {previous_verified+completed}/{len(rows)})")
        shutil.rmtree(local_stage)
        say(f"{group}: all downloaded windows committed and verified on Global Volume.")
    say("\nFINAL PASS: rehashing ALL selected files directly from the mounted Global Volume...")
    final_counts = {}
    for group, rows in records.items():
        for idx, row in enumerate(rows, 1):
            if not verified_path(dest_base / group / row["path"], row):
                raise RuntimeError(f"FINAL VALIDATION FAILED: {group}/{row['path']}")
            if idx % 500 == 0 or idx == len(rows):
                say(f"  {group}: {idx}/{len(rows)} files independently rehashed")
        final_counts[group] = len(rows)
    report = {
        "schema": "hephaestus-hf-global-restore-v1",
        "export_manifest_sha256": manifest_sha,
        "source_volume_id": EXPECTED_SOURCE_VOLUME,
        "global_volume_id": GLOBAL_VOLUME_ID,
        "hf_repository_revisions": revisions,
        "verified_counts": final_counts,
        "verified_files": TOTAL_FILES,
        "verified_bytes": total,
        "verification": "source SHA-256 in EXPORT_MANIFEST.json vs full downloaded file and Global Volume read-back",
        "source_export_scope": "selected curated files only; excludes third-party caches, mutable state, protected eval and other omitted paths",
        "note": "This receipt is not model certification, artifact policy admission, or proof of cross-region model loading.",
    }
    receipt_bytes = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    receipt = dest_base / "RESTORE_RECEIPT.json"
    if receipt.exists():
        if receipt.is_symlink() or receipt.read_bytes() != receipt_bytes:
            raise RuntimeError("Existing restore receipt differs; refusing overwrite")
    else:
        with receipt.open("xb") as f:
            f.write(receipt_bytes)
    if receipt.read_bytes() != receipt_bytes:
        raise RuntimeError("Global Volume restore receipt read-back failed")
    say("\nRESTORE_COMPLETE=true")
    say(f"VERIFIED_FILES={TOTAL_FILES}")
    say(f"VERIFIED_BYTES={total}")
    say(f"EXPORT_MANIFEST_SHA256={manifest_sha}")
    say(f"GLOBAL_RECEIPT={receipt}")
    say("Original Czech volume remains untouched; excluded files are NOT yet migrated.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        say("\nInterrupted. Re-run the same script on this Pod to resume. No source data was deleted.")
        raise SystemExit(130)
    except Exception as exc:
        # Avoid printing HTTP request headers and access tokens.
        say(f"\nSTOPPED: {type(exc).__name__}: {str(exc)[:350]}")
        say("No files were removed from the Global Volume. Resolve the error, then rerun the script.")
        raise SystemExit(1)