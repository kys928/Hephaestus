#!/usr/bin/env python3
"""Availability-priority, attempt-aware wrapper for second controlled training."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import launch_second_controlled_training as base
import launch_second_controlled_training_v2  # noqa: F401 - reviewed Pod runtime
from runpod_capacity_selection import create_with_capacity_retries

_capacity_observations: list[dict[str, object]] = []
_stale_execution_archive: dict[str, object] | None = None
_original_pod_shell = base.pod_shell


def _hash_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _execution_key(name: str) -> str:
    return f"{base.SCIENTIFIC_PREFIX}/executions/{base.RUN_ID}/{name}"


def _archive_stale_execution_evidence() -> dict[str, object] | None:
    """Preserve stale canonical execution files before starting a new attempt."""
    shared = base.base
    client = shared.s3_client()
    result_key = _execution_key("driver_result.json")
    result_raw = shared.maybe_read_key(client, result_key)
    log_key = _execution_key("pod_runtime.log")
    log_raw = shared.maybe_read_key(client, log_key)
    if result_raw is None and log_raw is None:
        return None

    current_sha = os.environ.get("GITHUB_SHA", "").strip()
    prior_sha = "unknown"
    prior_payload: dict[str, object] | None = None
    if result_raw is not None:
        parsed = json.loads(result_raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError("stale controlled-training sentinel is not a JSON object")
        prior_payload = parsed
        prior_sha = str(parsed.get("repo_sha") or "unknown").strip() or "unknown"
        if current_sha and prior_sha == current_sha:
            raise RuntimeError("current-commit controlled-training sentinel already exists; refusing ambiguous replay")

    identity_source = result_raw if result_raw is not None else log_raw or b""
    identity = hashlib.sha256(identity_source).hexdigest()[:16]
    archive_prefix = (
        f"{base.SCIENTIFIC_PREFIX}/failed_attempts/{base.RUN_ID}/execution-attempts/"
        f"{prior_sha[:16]}-{identity}"
    )
    copied: dict[str, dict[str, object]] = {}
    for name, source_key, raw in (
        ("driver_result.json", result_key, result_raw),
        ("pod_runtime.log", log_key, log_raw),
    ):
        if raw is None:
            continue
        destination = f"{archive_prefix}/{name}"
        if shared.maybe_read_key(client, destination) is not None:
            raise RuntimeError(f"stale-execution archive destination already exists: {destination}")
        client.put_object(Bucket=base.VOLUME_ID, Key=destination, Body=raw)
        roundtrip = shared.read_key(client, destination)
        if roundtrip != raw:
            raise RuntimeError(f"stale-execution archive round-trip mismatch: {destination}")
        copied[name] = {
            "source_key": source_key,
            "archive_key": destination,
            "bytes": len(raw),
            "sha256": _hash_bytes(raw),
        }

    manifest = {
        "archive_version": "controlled-training-execution-attempt.v1",
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "run_id": base.RUN_ID,
        "current_repo_sha": current_sha,
        "prior_repo_sha": prior_sha,
        "prior_status": prior_payload.get("status") if prior_payload else None,
        "files": copied,
    }
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_key = f"{archive_prefix}/archive_manifest.json"
    client.put_object(Bucket=base.VOLUME_ID, Key=manifest_key, Body=manifest_raw)
    if shared.read_key(client, manifest_key) != manifest_raw:
        raise RuntimeError("stale-execution archive manifest round-trip mismatch")

    # Delete canonical stale evidence only after every archive copy verifies.
    for _name, source_key, raw in (
        ("driver_result.json", result_key, result_raw),
        ("pod_runtime.log", log_key, log_raw),
    ):
        if raw is not None:
            client.delete_object(Bucket=base.VOLUME_ID, Key=source_key)
            if shared.maybe_read_key(client, source_key) is not None:
                raise RuntimeError(f"failed to clear stale canonical execution evidence: {source_key}")
    manifest["manifest_key"] = manifest_key
    return manifest


def _pod_shell_v3() -> str:
    shell = _original_pod_shell()
    shell = shell.replace(
        "scripts/run_second_controlled_training.py scripts/run_second_controlled_training_v2.py",
        "scripts/run_second_controlled_training.py scripts/run_second_controlled_training_v2.py scripts/run_second_controlled_training_v3.py",
    )
    shell = shell.replace(
        '"$PY" scripts/run_second_controlled_training_v2.py',
        '"$PY" scripts/run_second_controlled_training_v3.py',
    )
    if "run_second_controlled_training_v3.py" not in shell:
        raise RuntimeError("failed to project audited v3 controlled-training Pod command")
    return shell


def _create_capacity_aware(execution, repo_sha):
    global _capacity_observations

    def create_once(gpu_ids):
        body = {
            "name": f"hephaestus-{base.RUN_ID}"[:180],
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": list(gpu_ids),
            "gpuTypePriority": "availability",
            "cloudType": "SECURE",
            "dataCenterIds": [base.DATACENTER_ID],
            "dataCenterPriority": "custom",
            "imageName": base.IMAGE,
            "containerDiskInGb": 20,
            "networkVolumeId": base.VOLUME_ID,
            "volumeMountPath": "/workspace",
            "dockerStartCmd": ["bash", "-lc", base.pod_shell()],
            "interruptible": False,
            "env": {
                "HEPHAESTUS_RUN_ID": base.RUN_ID,
                "HEPHAESTUS_REPO_SHA": repo_sha,
                "HEPHAESTUS_OPERATOR_APPROVAL_REF": base.APPROVAL_REF,
                "HEPHAESTUS_MAX_WALL_SECONDS": "1200",
            },
        }
        return execution._create_pod(body)  # integration-only scheduler projection

    pod, observations = create_with_capacity_retries(create_once)
    _capacity_observations = observations
    return pod


def _wait_for_current_result(client, execution, pod_id):
    """Accept only a terminal sentinel produced by this exact launch commit."""
    shared = base.base
    expected_sha = os.environ.get("GITHUB_SHA", "").strip()
    if not expected_sha:
        raise RuntimeError("GITHUB_SHA is required for attempt-aware result verification")
    key = _execution_key("driver_result.json")
    deadline = time.monotonic() + base.MAX_SECONDS
    observations: list[dict[str, object]] = []
    last_status: str | None = None
    stale_hashes: set[str] = set()
    while time.monotonic() < deadline:
        raw = shared.maybe_read_key(client, key)
        if raw is not None:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("controlled training terminal record is not a JSON object")
            observed_sha = str(payload.get("repo_sha") or "").strip()
            if observed_sha == expected_sha:
                return payload, observations
            digest = _hash_bytes(raw)
            if digest not in stale_hashes:
                observations.append(
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "ignored_stale_terminal_sha256": digest,
                        "observed_repo_sha": observed_sha,
                        "expected_repo_sha": expected_sha,
                    }
                )
                stale_hashes.add(digest)
        pod = shared.pod_snapshot(execution, pod_id)
        status = str(pod.get("desiredStatus", "unknown")) if pod else "unknown"
        if status != last_status:
            observations.append({"at": datetime.now(timezone.utc).isoformat(), "desired_status": status})
            last_status = status
        time.sleep(base.POLL_SECONDS)
    raise TimeoutError(f"current controlled training result did not appear within {base.MAX_SECONDS} seconds")


base.pod_shell = _pod_shell_v3
base.create_pod = _create_capacity_aware
base.wait_for_result = _wait_for_current_result


if __name__ == "__main__":
    _stale_execution_archive = _archive_stale_execution_evidence()
    code = base.main()
    path = Path("second_controlled_training_launcher.json")
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["capacity_selection"] = _capacity_observations
        payload["training_driver"] = "scripts/run_second_controlled_training_v3.py"
        payload["stale_execution_archive"] = _stale_execution_archive
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(code)
