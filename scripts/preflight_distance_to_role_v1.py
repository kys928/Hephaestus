#!/usr/bin/env python3
"""CPU/S3 admission gate for Hephaestus distance-to-role V1."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_positive_promotion_proof as launcher
import run_distance_to_role_v1 as dtr

OUT = Path("distance_to_role_preflight")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_s3(s3: Any, key: str) -> bytes:
    return s3.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read()


def main() -> int:
    repo_sha = os.environ.get("GITHUB_SHA", "").strip()
    if not repo_sha:
        raise RuntimeError("GITHUB_SHA is required")
    protocol_raw = dtr.PROTOCOL_PATH.read_bytes()
    topology_raw = dtr.TOPOLOGY_PATH.read_bytes()
    semantic_raw = dtr.SEMANTIC_PATH.read_bytes()
    elasticity_protocol_raw = dtr.ELASTICITY_PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_raw)
    topology = json.loads(topology_raw)
    semantic = json.loads(semantic_raw)
    elasticity_protocol = json.loads(elasticity_protocol_raw)
    dtr.validate_protocol(protocol, topology, semantic)

    shared_training_keys = (
        "method", "peft_version", "precision", "rank", "alpha", "dropout", "bias",
        "target_module_suffixes", "learning_rate", "weight_decay", "micro_batch_size",
        "gradient_accumulation_steps", "max_seq_length", "warmup_ratio", "max_grad_norm",
        "optimizer", "shuffle_seed", "model_seed", "gradient_checkpointing", "train_only_assistant_tokens",
    )
    drift = {
        key: {"elasticity": elasticity_protocol["training"].get(key), "distance": protocol["training"].get(key)}
        for key in shared_training_keys
        if elasticity_protocol["training"].get(key) != protocol["training"].get(key)
    }
    if drift:
        raise RuntimeError(f"distance fixed training variables drifted from Test 2: {drift}")
    if sha(topology_raw) != protocol["sources"]["topology_protocol_sha256"]:
        raise RuntimeError("topology bytes drifted")
    if sha(elasticity_protocol_raw) != protocol["sources"]["elasticity_protocol_sha256"]:
        raise RuntimeError("elasticity protocol bytes drifted")
    if semantic.get("content_hash") != protocol["sources"]["semantic_content_hash"]:
        raise RuntimeError("semantic content hash drifted")

    candidate_ids = [row["model_id"] for row in topology["candidates"]]
    shards = protocol["execution"]["model_shards"]
    flattened = [*shards["olmo-granite"], *shards["qwen-ministral"]]
    if flattened != candidate_ids or len(set(flattened)) != len(candidate_ids):
        raise RuntimeError("distance shards do not exactly cover frozen candidates")

    s3 = launcher.base.s3_client()
    elasticity_key = f"{launcher.SCIENTIFIC_PREFIX}/adaptation_elasticity/{protocol['sources']['elasticity_run_id']}/elasticity_result.json"
    elasticity_raw = read_s3(s3, elasticity_key)
    if sha(elasticity_raw) != protocol["sources"]["elasticity_result_sha256"]:
        raise RuntimeError("source elasticity result hash mismatch")
    elasticity_result = json.loads(elasticity_raw)
    if elasticity_result.get("status") != "completed" or elasticity_result.get("disposition") != "scientific_elasticity_complete":
        raise RuntimeError("source elasticity result is incomplete")
    if elasticity_result.get("dataset_sha256") != protocol["sources"]["training_dataset_sha256"]:
        raise RuntimeError("source elasticity dataset hash mismatch")
    expected_models = [(row["model_id"], row["revision"]) for row in topology["candidates"]]
    observed_models = [(row.get("model_id"), row.get("revision")) for row in elasticity_result.get("model_results", [])]
    if observed_models != expected_models:
        raise RuntimeError("source elasticity candidate identities drifted")
    for model in elasticity_result["model_results"]:
        if model.get("status") != "complete" or set(model.get("roles", {})) != set(protocol["roles"]):
            raise RuntimeError("source elasticity role evidence is incomplete")

    source_sha = protocol["sources"]["elasticity_admission_repo_sha"]
    source_prefix = f"{launcher.SCIENTIFIC_PREFIX}/adaptation_elasticity/model_admission/{source_sha}"
    source_admission_raw = read_s3(s3, f"{source_prefix}/admission.json")
    dataset_raw = read_s3(s3, f"{source_prefix}/training.jsonl")
    contamination_raw = read_s3(s3, f"{source_prefix}/contamination_report.json")
    requirements_raw = read_s3(s3, f"{source_prefix}/requirements.txt")
    source_admission = json.loads(source_admission_raw)
    contamination = json.loads(contamination_raw)
    dataset_sha = sha(dataset_raw)
    if source_admission.get("repo_sha") != source_sha:
        raise RuntimeError("source adaptation admission repository identity drifted")
    if dataset_sha != protocol["sources"]["training_dataset_sha256"]:
        raise RuntimeError("source adaptation dataset differs from frozen distance source")
    if source_admission.get("dataset_manifest", {}).get("dataset_sha256") != dataset_sha:
        raise RuntimeError("source adaptation dataset manifest mismatch")
    if contamination.get("passed") is not True or contamination.get("dataset_sha256") != dataset_sha:
        raise RuntimeError("source adaptation contamination evidence is invalid")
    if sha(requirements_raw) != source_admission.get("requirements_sha256"):
        raise RuntimeError("source adaptation dependency lock hash mismatch")
    if source_admission.get("training_approved") is not True or source_admission.get("promotion_allowed") is not False or source_admission.get("lineage_mutation_allowed") is not False:
        raise RuntimeError("source adaptation governance is invalid")
    if source_admission.get("candidate_order") != candidate_ids:
        raise RuntimeError("source adaptation candidate order drifted")
    admitted_pairs = [(row.get("model_id"), row.get("revision")) for row in source_admission.get("candidates", [])]
    if admitted_pairs != expected_models:
        raise RuntimeError(f"source immutable candidate revisions drifted: {admitted_pairs}")
    token_lengths = source_admission.get("token_length_evidence", {})
    for model_id in candidate_ids:
        evidence = token_lengths.get(model_id, {})
        if int(evidence.get("max_training_tokens", 10**9)) > int(protocol["training"]["max_seq_length"]):
            raise RuntimeError(f"source training example exceeds distance max sequence length: {model_id}")

    dataset_rows = [json.loads(line) for line in dataset_raw.decode("utf-8").splitlines() if line.strip()]
    role_counts = {role: sum(1 for row in dataset_rows if row.get("role") == role) for role in protocol["roles"]}
    expected_counts = {role: int(protocol["training"]["examples_per_role"]) for role in protocol["roles"]}
    if role_counts != expected_counts:
        raise RuntimeError(f"distance source dataset role counts drifted: {role_counts}")

    train = protocol["training"]
    if int(train["examples_per_role"]) % int(train["gradient_accumulation_steps"]) != 0:
        raise RuntimeError("distance optimizer-step geometry is not exact")
    updates = int(train["examples_per_role"]) // int(train["gradient_accumulation_steps"])
    if updates != int(train["expected_optimizer_steps_per_epoch"]):
        raise RuntimeError("distance optimizer steps per epoch drifted")
    if max(train["dose_optimizer_steps"]) != updates * int(train["max_epochs"]):
        raise RuntimeError("distance maximum dose does not equal max epochs")

    admission = {
        "admission_version": "distance-to-role-admission.v1",
        "status": "admitted_execution_pending",
        "repo_sha": repo_sha,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha(protocol_raw),
        "topology_protocol_sha256": sha(topology_raw),
        "elasticity_protocol_sha256": sha(elasticity_protocol_raw),
        "elasticity_result_sha256": sha(elasticity_raw),
        "training_dataset_sha256": dataset_sha,
        "semantic_content_hash": semantic["content_hash"],
        "source_elasticity_key": elasticity_key,
        "source_adaptation_admission_prefix": source_prefix,
        "source_adaptation_admission_sha256": sha(source_admission_raw),
        "source_contamination_sha256": sha(contamination_raw),
        "requirements_sha256": sha(requirements_raw),
        "candidate_order": candidate_ids,
        "roles": protocol["roles"],
        "dose_optimizer_steps": train["dose_optimizer_steps"],
        "training_approved": True,
        "approval_source": protocol["governance"]["approval_source"],
        "promotion_allowed": False,
        "lineage_mutation_allowed": False,
        "frozen_eval_mutation_allowed": False,
        "persistent_model_cache_allowed": False
    }

    OUT.mkdir(exist_ok=True)
    objects = {
        "admission.json": (json.dumps(admission, indent=2, sort_keys=True) + "\n").encode(),
        "protocol.json": protocol_raw,
        "training.jsonl": dataset_raw,
        "contamination_report.json": contamination_raw,
        "source_model_admission.json": source_admission_raw,
        "requirements.txt": requirements_raw
    }
    for name, payload in objects.items():
        (OUT / name).write_bytes(payload)

    prefix = f"{launcher.SCIENTIFIC_PREFIX}/distance_to_role/model_admission/{repo_sha}"
    for name, payload in objects.items():
        key = f"{prefix}/{name}"
        s3.put_object(Bucket=launcher.VOLUME_ID, Key=key, Body=payload)
        if read_s3(s3, key) != payload:
            raise RuntimeError(f"distance admission S3 readback mismatch: {key}")
        print("DISTANCE_ADMISSION_PERSISTED_JSON " + json.dumps({"key": key, "sha256": sha(payload), "bytes": len(payload)}, sort_keys=True))

    print("DISTANCE_PREFLIGHT_JSON " + json.dumps({
        "status": "admitted_execution_pending", "protocol_id": protocol["protocol_id"], "protocol_sha256": sha(protocol_raw),
        "dataset_sha256": dataset_sha, "source_elasticity_result_sha256": sha(elasticity_raw),
        "candidate_count": len(candidate_ids), "role_count": len(protocol["roles"]), "dose_count": len(train["dose_optimizer_steps"])
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
