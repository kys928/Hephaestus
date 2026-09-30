#!/usr/bin/env python3
"""CPU/S3 admission gate for Hephaestus adaptation-elasticity V1."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_positive_promotion_proof as launcher
from build_adaptation_elasticity_dataset_v1 import build_dataset, contamination_report, training_user_prompt
from hephaestus.providers.models.admission import inspect_immutable_model

PROTOCOL_PATH = Path("configs/experiments/hephaestus_adaptation_elasticity_v1.json")
TOPOLOGY_PATH = Path("configs/eval_packs/hephaestus_cognitive_topology_v1.json")
OUT = Path("adaptation_elasticity_preflight")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _get_s3_json(s3, key: str) -> dict[str, Any]:
    raw = s3.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read()
    return json.loads(raw)


def _validate_protocol(protocol: dict[str, Any], topology: dict[str, Any], topology_sha: str) -> None:
    if protocol.get("protocol_id") != "hephaestus_adaptation_elasticity_v1":
        raise ValueError("unexpected adaptation elasticity protocol id")
    if protocol.get("scientific_variable") != "base_model_identity_under_fixed_role_lora_dose":
        raise ValueError("adaptation elasticity scientific variable drifted")
    baseline = protocol["baseline"]
    if baseline["protocol_id"] != topology["protocol_id"] or baseline["protocol_sha256"] != topology_sha:
        raise ValueError("adaptation baseline does not point to the frozen topology protocol")
    if protocol["roles"] != topology["roles"]:
        raise ValueError("adaptation roles drifted from topology V1")
    train = protocol["training"]
    required = {"method": "lora", "peft_version": "0.20.0", "precision": "bfloat16", "rank": 8, "alpha": 16, "dropout": 0.0, "learning_rate": 0.00005, "epochs": 2, "micro_batch_size": 1, "gradient_accumulation_steps": 4, "max_seq_length": 1024, "dose_checkpoints": [1, 2]}
    for key, expected in required.items():
        if train.get(key) != expected:
            raise ValueError(f"frozen low-dose training variable drifted: {key}")
    gov = protocol["governance"]
    if not gov.get("training_is_experimentally_approved") or gov.get("promotion_allowed") or gov.get("lineage_mutation_allowed"):
        raise ValueError("adaptation governance boundary is invalid")
    if not gov.get("dataset_must_pass_contamination_gate") or gov.get("frozen_eval_mutation_allowed"):
        raise ValueError("adaptation evaluation/data governance boundary is invalid")


def _write_dataset(protocol: dict[str, Any], topology: dict[str, Any]) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    rows = build_dataset()
    expected = int(protocol["dataset"]["examples_per_role"])
    counts = {role: sum(1 for row in rows if row["role"] == role) for role in protocol["roles"]}
    if any(count != expected for count in counts.values()):
        raise RuntimeError(f"unexpected role counts: {counts}")
    report = contamination_report(rows, topology, protocol)
    if not report["passed"]:
        raise RuntimeError(f"training/eval contamination gate failed: {report['violations'][:5]}")
    dataset_raw = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode()
    report["dataset_sha256"] = _sha(dataset_raw)
    report["role_counts"] = counts
    report_raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    manifest = {"dataset_id": protocol["dataset"]["generator_id"], "dataset_sha256": report["dataset_sha256"], "record_count": len(rows), "role_counts": counts, "source": protocol["dataset"]["source"], "approval_source": protocol["dataset"]["approval_source"], "approval_scope": protocol["protocol_id"], "contamination_passed": True, "frozen_eval_prompt_or_evidence_identifiers_present": False}
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    return dataset_raw, report_raw, manifest_raw, manifest


def _validate_token_lengths(topology: dict[str, Any], protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    max_len = int(protocol["training"]["max_seq_length"])
    results: dict[str, Any] = {}
    for candidate in topology["candidates"]:
        tokenizer = AutoTokenizer.from_pretrained(candidate["model_id"], revision=candidate["revision"], trust_remote_code=False)
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError(f"candidate lacks native chat template: {candidate['model_id']}")
        observed_max = 0
        for row in rows:
            user = training_user_prompt(row)
            target = json.dumps(row["target"], separators=(",", ":"), ensure_ascii=False)
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": target}], tokenize=False, add_generation_prompt=False)
            length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            observed_max = max(observed_max, length)
            if length > max_len:
                raise RuntimeError(f"training example exceeds max_seq_length for {candidate['model_id']}: {length}>{max_len}")
        results[candidate["model_id"]] = {"max_training_tokens": observed_max, "max_seq_length": max_len}
    return results


def main() -> int:
    protocol_raw = PROTOCOL_PATH.read_bytes()
    topology_raw = TOPOLOGY_PATH.read_bytes()
    protocol = json.loads(protocol_raw)
    topology = json.loads(topology_raw)
    protocol_sha = _sha(protocol_raw)
    topology_sha = _sha(topology_raw)
    _validate_protocol(protocol, topology, topology_sha)
    OUT.mkdir(exist_ok=True)

    rows = build_dataset()
    dataset_raw, contamination_raw, dataset_manifest_raw, dataset_manifest = _write_dataset(protocol, topology)
    (OUT / "training.jsonl").write_bytes(dataset_raw)
    (OUT / "contamination_report.json").write_bytes(contamination_raw)
    (OUT / "dataset_manifest.json").write_bytes(dataset_manifest_raw)

    s3 = launcher.base.s3_client()
    baseline = protocol["baseline"]
    baseline_key = f"{launcher.SCIENTIFIC_PREFIX}/cognitive_topology/{baseline['run_id']}/cohort_result.json"
    baseline_result = _get_s3_json(s3, baseline_key)
    if baseline_result.get("status") != "completed" or baseline_result.get("protocol_sha256") != topology_sha:
        raise RuntimeError("frozen topology baseline is unavailable or mismatched")
    expected_ids = [(row["model_id"], row["revision"]) for row in topology["candidates"]]
    observed_ids = [(row["model_id"], row["revision"]) for row in baseline_result.get("model_summaries", [])]
    if observed_ids != expected_ids:
        raise RuntimeError("baseline candidate identities drifted")

    records = []
    for candidate in topology["candidates"]:
        record = inspect_immutable_model(candidate, OUT / "metadata_cache")
        records.append(record)
        print("ELASTICITY_MODEL_ADMISSION_JSON " + json.dumps(record.to_dict(), sort_keys=True))

    token_length_evidence = _validate_token_lengths(topology, protocol, rows)
    packages = ["transformers", "accelerate", "tokenizers", "safetensors", "huggingface-hub", "peft"]
    locked = {name: importlib.metadata.version(name) for name in packages}
    if locked["peft"] != protocol["training"]["peft_version"]:
        raise RuntimeError("PEFT runtime version differs from frozen protocol")
    lock = "".join(f"{name}=={version}\n" for name, version in locked.items())

    admission = {"admission_version": "adaptation-elasticity-admission.v1", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha, "topology_protocol_id": topology["protocol_id"], "topology_protocol_sha256": topology_sha, "baseline_run_id": baseline["run_id"], "repo_sha": os.environ["GITHUB_SHA"], "status": "metadata_dataset_admitted_training_pending", "scientific_variable": protocol["scientific_variable"], "candidate_order": [candidate["model_id"] for candidate in topology["candidates"]], "candidates": [record.to_dict() for record in records], "runtime_packages": locked, "requirements_sha256": _sha(lock.encode()), "dataset_manifest": dataset_manifest, "contamination_report_sha256": _sha(contamination_raw), "token_length_evidence": token_length_evidence, "training_approved": True, "training_approval_source": protocol["dataset"]["approval_source"], "lineage_mutation_allowed": False, "promotion_allowed": False, "frozen_eval_mutation_allowed": False}
    admission_raw = (json.dumps(admission, indent=2, sort_keys=True) + "\n").encode()
    requirements_raw = lock.encode()
    (OUT / "admission.json").write_bytes(admission_raw)
    (OUT / "requirements.txt").write_bytes(requirements_raw)

    prefix = f"{launcher.SCIENTIFIC_PREFIX}/adaptation_elasticity/model_admission/{os.environ['GITHUB_SHA']}"
    objects = {"admission.json": admission_raw, "requirements.txt": requirements_raw, "training.jsonl": dataset_raw, "contamination_report.json": contamination_raw, "dataset_manifest.json": dataset_manifest_raw, "protocol.json": protocol_raw}
    for name, payload in objects.items():
        key = f"{prefix}/{name}"
        s3.put_object(Bucket=launcher.VOLUME_ID, Key=key, Body=payload)
        observed = s3.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read()
        if observed != payload:
            raise RuntimeError(f"S3 adaptation admission readback mismatch: {key}")
        print("ELASTICITY_ADMISSION_PERSISTED_JSON " + json.dumps({"key": key, "sha256": _sha(payload), "bytes": len(payload)}, sort_keys=True))

    print("ELASTICITY_PROTOCOL_JSON " + json.dumps({"protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha, "topology_protocol_sha256": topology_sha, "dataset_sha256": dataset_manifest["dataset_sha256"], "candidate_count": len(topology["candidates"]), "role_count": len(protocol["roles"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
