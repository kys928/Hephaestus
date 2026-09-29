#!/usr/bin/env python3
"""Launch Interface Repair V2 full stack from a successful bound preflight.

The preflight necessarily runs on an earlier commit than the full-launch authorization
commit. This wrapper permits that child commit only when every change since the
preflight is authorization-only and the frozen scientific/execution configuration is
otherwise byte-for-byte equivalent after normalizing the two paid gate booleans.
"""
from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import launch_interface_repair_v2 as launcher

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_repair_v2.json"
MARKER_PATH = ROOT / "configs/experiments/interface_repair_v2.launch.json"
ALLOWED_CHANGED_FILES = {
    "configs/experiments/hephaestus_interface_repair_v2.json",
    "configs/experiments/interface_repair_v2.launch.json",
}
ALLOWED_GOVERNANCE_KEYS = {"paid_preflight_allowed", "paid_full_launch_allowed"}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _json_at(sha: str, path: str) -> dict[str, Any]:
    raw = _git("show", f"{sha}:{path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}@{sha} is not a JSON object")
    return value


def _normalized_config(cfg: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(cfg)
    governance = value.get("governance")
    if not isinstance(governance, dict):
        raise RuntimeError("V2 config governance must be an object")
    for key in ALLOWED_GOVERNANCE_KEYS:
        governance.pop(key, None)
    return value


def verify_authorization_only_child(bound_sha: str, current_sha: str, current_cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", bound_sha, current_sha],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("full-launch commit is not a descendant of the successful preflight commit") from exc

    changed = {line for line in _git("diff", "--name-only", f"{bound_sha}..{current_sha}").splitlines() if line}
    unexpected = changed - ALLOWED_CHANGED_FILES
    if unexpected:
        raise RuntimeError(f"non-authorization files changed since preflight: {sorted(unexpected)}")

    preflight_cfg = _json_at(bound_sha, "configs/experiments/hephaestus_interface_repair_v2.json")
    if _normalized_config(preflight_cfg) != _normalized_config(current_cfg):
        raise RuntimeError("scientific or execution configuration changed after preflight")

    return {
        "preflight_repo_sha": bound_sha,
        "full_repo_sha": current_sha,
        "changed_files": sorted(changed),
        "authorization_only": True,
    }


def verified_preflight(client: Any, marker: dict[str, Any], current_sha: str, cfg: dict[str, Any]) -> dict[str, Any]:
    key = str(marker.get("preflight_evidence_key") or "")
    bound_sha = str(marker.get("preflight_repo_sha") or "")
    if not key or not bound_sha:
        raise RuntimeError("full launch requires persisted preflight evidence and preflight repository SHA")

    binding = verify_authorization_only_child(bound_sha, current_sha, cfg)
    result = launcher.maybe_json(client, key)
    if result is None:
        raise RuntimeError("bound preflight evidence does not exist")
    if result.get("status") != "completed" or result.get("approved_for_full_run") is not True:
        raise RuntimeError("bound preflight did not approve a full run")
    if str(result.get("repo_sha", "")) != bound_sha:
        raise RuntimeError("preflight evidence repository SHA does not match the bound preflight commit")

    print("INTERFACE_REPAIR_V2_PREFLIGHT_BINDING_JSON " + json.dumps(binding, sort_keys=True), flush=True)
    return result


def main() -> int:
    cfg = launcher.load_json(CFG_PATH)
    marker = launcher.load_json(MARKER_PATH)
    launcher.authorize(cfg, marker, "full")
    current_sha = launcher.required("GITHUB_SHA")
    client = launcher.s3_client()
    client.head_bucket(Bucket=launcher.required("RUNPOD_NETWORK_VOLUME_ID"))
    evidence = verified_preflight(client, marker, current_sha, cfg)

    # execute() performs all normal cost, lifecycle, teardown, and governance checks.
    # Replace only the impossible exact-SHA verifier with the already-validated
    # authorization-only-child evidence result above.
    launcher.verify_preflight_evidence = lambda _client, _marker, _sha: evidence
    result = launcher.execute(cfg, marker, "full")
    print("INTERFACE_REPAIR_V2_LAUNCH_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
