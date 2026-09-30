#!/usr/bin/env python3
"""CPU/S3 admission gate for the non-mutating Hephaestus cognitive topology cohort."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from hephaestus.providers.models.admission import inspect_immutable_model

import launch_positive_promotion_proof as launcher

SPEC_PATH = Path("configs/eval_packs/hephaestus_cognitive_topology_v1.json")
OUT = Path("cognitive_topology_preflight")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_spec(spec: dict[str, object]) -> None:
    if spec.get("protocol_id") != "hephaestus_cognitive_topology_v1":
        raise ValueError("unexpected cognitive-topology protocol id")
    if spec.get("scientific_variable") != "candidate_model_identity":
        raise ValueError("candidate model identity must be the only scientific variable")
    if spec.get("non_mutating") is not True:
        raise ValueError("cognitive topology cohort must be non-mutating")
    candidates = spec.get("candidates")
    cases = spec.get("cases")
    roles = spec.get("roles")
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError("topology cohort requires exactly four candidates")
    if not isinstance(cases, list) or len(cases) < 25:
        raise ValueError("topology cohort is too small to support the requested role analysis")
    if roles != ["diagnosis", "planner", "evaluator", "judge", "controller"]:
        raise ValueError("topology role order drifted")
    weights = dict(spec.get("scoring", {}))
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("topology scoring weights must sum to one")
    case_ids: set[str] = set()
    role_counts = {role: 0 for role in roles}
    required_keys = set(dict(spec["response_schema"])["required_exact_keys"])
    if required_keys != {"decision", "action", "primary_variable", "confidence", "evidence_refs", "uncertainties", "rationale"}:
        raise ValueError("response schema drifted")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        case_id = str(case.get("case_id", ""))
        role = str(case.get("role", ""))
        if not case_id or case_id in case_ids or role not in role_counts:
            raise ValueError(f"invalid or duplicate case: {case_id!r}")
        case_ids.add(case_id)
        role_counts[role] += 1
        evidence = case.get("evidence")
        allowed = case.get("allowed_evidence_refs")
        expected = case.get("expected")
        if not isinstance(evidence, list) or not evidence or not isinstance(allowed, list) or not isinstance(expected, dict):
            raise ValueError(f"case {case_id} lacks governed evidence/expectation")
        refs = {str(item.get("ref")) for item in evidence if isinstance(item, dict)}
        if not set(map(str, allowed)).issubset(refs):
            raise ValueError(f"case {case_id} allows an evidence ref absent from the prompt")
        for key in ("decision", "action", "primary_variable", "confidence_min", "confidence_max"):
            if key not in expected:
                raise ValueError(f"case {case_id} expected result lacks {key}")
        if not 0.0 <= float(expected["confidence_min"]) <= float(expected["confidence_max"]) <= 1.0:
            raise ValueError(f"case {case_id} confidence range is invalid")
    if any(count < 5 for count in role_counts.values()):
        raise ValueError(f"every role needs at least five cases: {role_counts}")
    conditions = {str(case.get("condition")) for case in cases}
    for required in ("irrelevant", "contradictory", "missing_evidence", "hard_regression", "forbidden"):
        if required not in conditions:
            raise ValueError(f"required adversarial condition absent: {required}")


def main() -> int:
    raw = SPEC_PATH.read_bytes()
    spec = json.loads(raw)
    _validate_spec(spec)
    protocol_sha256 = _sha256(raw)
    OUT.mkdir(exist_ok=True)

    records = []
    for candidate in spec["candidates"]:
        record = inspect_immutable_model(candidate, OUT / "metadata_cache")
        records.append(record)
        print("TOPOLOGY_MODEL_ADMISSION_JSON " + json.dumps(record.to_dict(), sort_keys=True))

    packages = ["transformers", "accelerate", "tokenizers", "safetensors", "huggingface-hub"]
    locked = {name: importlib.metadata.version(name) for name in packages}
    lock = "".join(f"{name}=={version}\n" for name, version in locked.items())
    admission = {
        "admission_version": "cognitive-topology-admission.v1",
        "protocol_id": spec["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "repo_sha": os.environ["GITHUB_SHA"],
        "status": "metadata_admitted_runtime_pending",
        "scientific_variable": spec["scientific_variable"],
        "candidate_order": [candidate["model_id"] for candidate in spec["candidates"]],
        "candidates": [record.to_dict() for record in records],
        "runtime_packages": locked,
        "requirements_sha256": _sha256(lock.encode()),
        "gpu_generation_tested": False,
        "lineage_mutation_allowed": False,
        "training_allowed": False,
        "promotion_allowed": False
    }
    admission_raw = (json.dumps(admission, indent=2, sort_keys=True) + "\n").encode()
    requirements_raw = lock.encode()
    (OUT / "admission.json").write_bytes(admission_raw)
    (OUT / "requirements.txt").write_bytes(requirements_raw)

    s3 = launcher.base.s3_client()
    prefix = f"{launcher.SCIENTIFIC_PREFIX}/cognitive_topology/model_admission/{os.environ['GITHUB_SHA']}"
    for name, payload in (("admission.json", admission_raw), ("requirements.txt", requirements_raw)):
        key = f"{prefix}/{name}"
        s3.put_object(Bucket=launcher.VOLUME_ID, Key=key, Body=payload)
        observed = s3.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read()
        if observed != payload:
            raise RuntimeError(f"S3 admission readback mismatch: {key}")
        print("TOPOLOGY_ADMISSION_PERSISTED_JSON " + json.dumps({
            "key": key, "sha256": _sha256(payload), "bytes": len(payload)
        }, sort_keys=True))
    print("TOPOLOGY_PROTOCOL_JSON " + json.dumps({
        "protocol_id": spec["protocol_id"], "protocol_sha256": protocol_sha256,
        "case_count": len(spec["cases"]), "candidate_count": len(spec["candidates"])
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
