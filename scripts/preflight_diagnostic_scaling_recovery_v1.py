#!/usr/bin/env python3
"""Validate Recovery V1 inputs without creating paid compute.

Default mode is repository/Hugging Face validation only. --verify-source-s3 performs
read-only verification of the already-existing Test-3 training bytes and
contamination evidence. --persist-launch-admission remains gated by the committed
recovery authorization state plus the second authorization environment lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_first_bounded_scientific_training as storage
import validate_diagnostic_scaling_recovery_v1 as validator

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/experiments/hephaestus_diagnostic_scaling_recovery_v1.json"
BASE_DISTANCE_PATH = ROOT / "configs/experiments/hephaestus_distance_to_role_v1.json"
OUT = ROOT / "diagnostic_scaling_recovery_preflight"
AUTH_ENV = "HEPHAESTUS_DIAGNOSTIC_SCALING_RECOVERY_LAUNCH_AUTHORIZED"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def requirements_bytes(contract: dict[str, Any]) -> bytes:
    deps = contract["execution"]["runtime_dependencies"]
    return ("\n".join([
        f"transformers=={deps['transformers']}",
        f"accelerate=={deps['accelerate']}",
        f"safetensors=={deps['safetensors']}",
        f"huggingface-hub=={deps['huggingface_hub']}",
        f"peft=={deps['peft']}",
        "boto3>=1.35,<2",
    ]) + "\n").encode("utf-8")


def source_keys() -> tuple[str, str]:
    base_distance = json.loads(BASE_DISTANCE_PATH.read_text(encoding="utf-8"))
    source_sha = base_distance["sources"]["elasticity_admission_repo_sha"]
    prefix = f"{storage.SCIENTIFIC_PREFIX}/adaptation_elasticity/model_admission/{source_sha}"
    return f"{prefix}/training.jsonl", f"{prefix}/contamination_report.json"


def verify_source_s3(contract: dict[str, Any]) -> dict[str, Any]:
    client = storage.s3_client()
    training_key, contamination_key = source_keys()
    training_raw = storage.read_key(client, training_key)
    contamination_raw = storage.read_key(client, contamination_key)
    dataset_sha = sha(training_raw)
    if dataset_sha != contract["sources"]["source_training_dataset_sha256"]:
        raise RuntimeError(f"source Test-3 training hash drifted: {dataset_sha}")
    contamination = json.loads(contamination_raw)
    if contamination.get("passed") is not True:
        raise RuntimeError("source Test-3 contamination evidence no longer passes")
    rows = [json.loads(line) for line in training_raw.decode("utf-8").splitlines() if line.strip()]
    roles = [*contract["roles"]["primary"], *contract["roles"]["calibration"]]
    expected = int(contract["training"]["examples_per_role"])
    role_counts = {role: sum(1 for row in rows if row.get("role") == role) for role in roles}
    if role_counts != {role: expected for role in roles}:
        raise RuntimeError(f"source Test-3 role counts drifted: {role_counts}")
    return {
        "verified": True,
        "training_key": training_key,
        "training_dataset_sha256": dataset_sha,
        "training_bytes": len(training_raw),
        "contamination_key": contamination_key,
        "contamination_sha256": sha(contamination_raw),
        "role_counts": role_counts,
        "training_raw": training_raw,
        "contamination_raw": contamination_raw,
    }


def build_preflight(*, repo_sha: str, verify_s3: bool) -> tuple[dict[str, Any], dict[str, bytes]]:
    contract_raw = CONTRACT_PATH.read_bytes()
    contract = json.loads(contract_raw)
    local = validator.validate_contract(contract, ROOT)
    remote = validator.validate_remote(contract)
    source = verify_source_s3(contract) if verify_s3 else {
        "verified": False,
        "reason": "source bytes intentionally not fetched in repository-only preflight",
        "training_dataset_sha256": contract["sources"]["source_training_dataset_sha256"],
    }
    requirements = requirements_bytes(contract)
    paid_allowed = bool(contract["governance"]["paid_launch_allowed_now"])
    summary = {
        "preflight_version": "diagnostic-scaling-recovery-preflight.v1",
        "status": "recovery_inputs_validated_launch_authorized" if paid_allowed else "recovery_inputs_validated_not_launchable",
        "repo_sha": repo_sha,
        "protocol_id": contract["protocol_id"],
        "protocol_sha256": sha(contract_raw),
        "candidate_revisions": {row["model_id"]: row["revision"] for row in contract["candidates"]},
        "fixed_non_thinking_eligible": local["fixed_non_thinking_eligible"],
        "reasoning_aware_eligible": local["reasoning_aware_eligible"],
        "authorization_state": local["authorization_state"],
        "remote_models": remote,
        "source_training": {key: value for key, value in source.items() if key not in {"training_raw", "contamination_raw"}},
        "runtime_requirements_sha256": sha(requirements),
        "paid_launch_allowed_now": paid_allowed,
        "paid_launch_admission_persisted": False,
        "network_volume_attached": False,
        "promotion_allowed": False,
        "lineage_mutation_allowed": False,
    }
    objects = {
        "preflight.json": (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "protocol.json": contract_raw,
        "requirements.txt": requirements,
        "remote_models.json": (json.dumps({"models": remote}, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    }
    if source.get("verified"):
        objects["training.jsonl"] = source["training_raw"]
        objects["contamination_report.json"] = source["contamination_raw"]
    return summary, objects


def persist_launch_admission(*, repo_sha: str, summary: dict[str, Any], objects: dict[str, bytes]) -> dict[str, Any]:
    contract = json.loads(objects["protocol.json"])
    if contract["governance"].get("paid_launch_allowed_now") is not True:
        raise RuntimeError("cannot persist a launchable Recovery admission while paid_launch_allowed_now is false")
    if contract["governance"].get("paid_launch_requires_new_explicit_user_go") is not True:
        raise RuntimeError("Recovery contract lost explicit user-go requirement")
    if os.environ.get(AUTH_ENV, "").strip() != "YES":
        raise RuntimeError(f"cannot persist launch admission without {AUTH_ENV}=YES")
    if "training.jsonl" not in objects or "contamination_report.json" not in objects:
        raise RuntimeError("launch admission requires independently re-read source training and contamination bytes")

    admission = {
        "admission_version": "diagnostic-scaling-recovery-admission.v1",
        "status": "recovery_launch_inputs_admitted",
        "repo_sha": repo_sha,
        "protocol_id": contract["protocol_id"],
        "protocol_sha256": sha(objects["protocol.json"]),
        "training_dataset_sha256": sha(objects["training.jsonl"]),
        "candidate_revisions": summary["candidate_revisions"],
        "fixed_non_thinking_eligible": summary["fixed_non_thinking_eligible"],
        "reasoning_aware_eligible": summary["reasoning_aware_eligible"],
        "paid_launch_authorized": True,
        "authorization_state": summary["authorization_state"],
        "network_volume_attached": False,
        "promotion_allowed": False,
        "lineage_mutation_allowed": False,
    }
    admission_raw = (json.dumps(admission, indent=2, sort_keys=True) + "\n").encode("utf-8")
    persist = dict(objects)
    persist["admission.json"] = admission_raw
    prefix = f"{storage.SCIENTIFIC_PREFIX}/diagnostic_scaling_recovery/model_admission/{repo_sha}"
    client = storage.s3_client()
    persisted = []
    for name, raw in persist.items():
        if name == "preflight.json":
            continue
        key = f"{prefix}/{name}"
        client.put_object(Bucket=storage.VOLUME_ID, Key=key, Body=raw)
        observed = storage.read_key(client, key)
        if observed != raw:
            raise RuntimeError(f"Recovery admission S3 readback mismatch: {key}")
        persisted.append({"key": key, "sha256": sha(raw), "bytes": len(raw)})
    return {"status": "recovery_launch_inputs_admitted", "prefix": prefix, "objects": persisted}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-source-s3", action="store_true", help="read-only verify existing Test-3 source bytes")
    parser.add_argument("--persist-launch-admission", action="store_true", help="persist launchable S3 admission; requires committed user authorization plus environment lock")
    args = parser.parse_args()
    repo_sha = os.environ.get("GITHUB_SHA", "local-preflight").strip()
    summary, objects = build_preflight(repo_sha=repo_sha, verify_s3=args.verify_source_s3 or args.persist_launch_admission)
    OUT.mkdir(exist_ok=True)
    for name, raw in objects.items():
        (OUT / name).write_bytes(raw)
    if args.persist_launch_admission:
        summary["launch_admission"] = persist_launch_admission(repo_sha=repo_sha, summary=summary, objects=objects)
    (OUT / "preflight_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DIAGNOSTIC_SCALING_RECOVERY_PREFLIGHT_JSON " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
