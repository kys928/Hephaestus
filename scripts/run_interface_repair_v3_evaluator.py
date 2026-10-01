#!/usr/bin/env python3
"""Evaluator-only Interface Repair V3 runner.

V3 continues from the persisted V2 Evaluator adapter, trains only the Evaluator on
fresh crossed/contrastive data, and separates three questions during certification:

1. Can the model infer the correct scientific state?
2. Can strict evidence-reference handling keep citations grounded without fuzzy repair?
3. Do deterministic Evaluator boundaries project state into the authoritative
   primary variable and role-local Hephaestus action?

The model's proposed action and primary variable are retained as telemetry but are
never authoritative system outputs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import run_interface_repair_v1 as v1
import run_interface_repair_v2 as v2
from interface_repair_v2_bootstrap import load_stack_compatible, materialize_parent_cached
from hephaestus.control.evaluator_boundary import (
    project_evaluator_action,
    project_evaluator_primary_variable,
)
from hephaestus.control.evaluator_evidence_refs import canonicalize_evaluator_evidence_refs

ROOT = v2.ROOT
CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_repair_v3_evaluator.json"
v2.CFG_PATH = CFG_PATH


def load_stack_v3(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Load the certified Phase-I base registry and replace only Evaluator adapter.

    The replacement adapter is the exact persisted V2 Evaluator artifact. It is
    intentionally labelled an uncertified diagnostic parent in the V3 config.
    """

    stack = load_stack_compatible(cfg)
    source = dict(cfg["source_evaluator_adapter"])
    if source.get("status") != "diagnostic_uncertified_v2_candidate":
        raise RuntimeError("V3 source Evaluator adapter must remain explicitly uncertified")
    evaluator = dict(stack["roles"]["evaluator"])
    if evaluator.get("model_id") != source.get("model_id") or evaluator.get("revision") != source.get("revision"):
        raise RuntimeError("V3 Evaluator base model/revision differs from certified Phase-I registry")
    evaluator["adapter"] = {
        "s3_key": source["s3_key"],
        "sha256": source["sha256"],
        "bytes": int(source["bytes"]),
    }
    evaluator["status"] = "v3_diagnostic_parent"
    roles = dict(stack["roles"])
    roles["evaluator"] = evaluator
    stack = dict(stack)
    stack["roles"] = roles
    return stack


def put_bytes_verified_compatible(client: Any, key: str, raw: bytes, content_type: str) -> dict[str, Any]:
    digest = hashlib.sha256(raw).hexdigest()
    bucket = v1.bucket()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=raw,
        ContentType=content_type,
        Metadata={"sha256": digest},
    )
    head = client.head_object(Bucket=bucket, Key=key)
    if int(head["ContentLength"]) != len(raw):
        raise RuntimeError(f"S3 size verification failed for {key}")
    observed_digest = str((head.get("Metadata") or {}).get("sha256", "")).strip()
    if observed_digest:
        if observed_digest != digest:
            raise RuntimeError(f"S3 metadata SHA-256 mismatch for {key}")
        return {"s3_key": key, "sha256": digest, "bytes": len(raw), "verification": "head_metadata"}
    response = client.get_object(Bucket=bucket, Key=key)
    try:
        observed = response["Body"].read()
    finally:
        response["Body"].close()
    if len(observed) != len(raw) or hashlib.sha256(observed).hexdigest() != digest:
        raise RuntimeError(f"S3 GET/SHA-256 verification failed for {key}")
    return {"s3_key": key, "sha256": digest, "bytes": len(raw), "verification": "get_sha256_fallback"}


def target_answer_v3(case: Mapping[str, Any]) -> str:
    expected = case["expected"]
    confidence = round((float(expected["confidence_min"]) + float(expected["confidence_max"])) / 2, 3)
    uncertainties: list[str] = []
    if str(expected["decision"]) in {"incomplete_evidence", "inconclusive", "recheck_required"}:
        uncertainties = ["A verified blocking condition remains unresolved; do not escalate scientific status."]
    payload = {
        "decision": expected["decision"],
        "action": expected["action"],
        "primary_variable": expected["primary_variable"],
        "confidence": confidence,
        "evidence_refs": list(case["allowed_evidence_refs"]),
        "uncertainties": uncertainties,
        "rationale": str(case["target_rationale"]),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _schema_valid(pack: Mapping[str, Any], obj: dict[str, Any] | None) -> bool:
    if obj is None:
        return False
    vocab = pack["contract_vocabulary"]["evaluator"]
    if set(obj) != set(pack["response_schema"]["required_exact_keys"]):
        return False
    if str(obj.get("decision")) not in set(vocab["decision"]):
        return False
    if str(obj.get("action")) not in set(vocab["action"]):
        return False
    if str(obj.get("primary_variable")) not in set(vocab["primary_variable"]):
        return False
    if not isinstance(obj.get("evidence_refs"), list) or not isinstance(obj.get("uncertainties"), list):
        return False
    if not isinstance(obj.get("rationale"), str):
        return False
    try:
        return 0 <= float(obj.get("confidence")) <= 1
    except Exception:
        return False


def score_case_v3(pack: Mapping[str, Any], case: Mapping[str, Any], role: str, raw: str) -> dict[str, Any]:
    if role != "evaluator":
        raise RuntimeError("V3 scorer is evaluator-only")
    obj, reason = v1.extract_json(raw)
    expected = case["expected"]
    schema = _schema_valid(pack, obj)

    decision = str(obj.get("decision", "")) if obj is not None else ""
    model_primary = str(obj.get("primary_variable", "")) if obj is not None else ""
    decision_exact = bool(obj is not None and decision == str(expected["decision"]))
    model_primary_exact = bool(obj is not None and model_primary == str(expected["primary_variable"]))
    primary_projection = project_evaluator_primary_variable(
        decision,
        model_proposed_primary_variable=model_primary,
    )
    primary_projected = not primary_projection.blocked and primary_projection.effective_primary_variable is not None
    primary_exact = bool(
        primary_projected
        and primary_projection.effective_primary_variable == str(expected["primary_variable"])
    )

    try:
        confidence = float(obj.get("confidence", -1)) if obj is not None else -1.0
    except Exception:
        confidence = -1.0
    confidence_ok = float(expected["confidence_min"]) <= confidence <= float(expected["confidence_max"])
    semantic_exact = decision_exact and primary_exact and confidence_ok
    model_action_agrees = bool(obj is not None and str(obj.get("action")) == str(expected["action"]))

    cited = [
        str(x) for x in (
            obj.get("evidence_refs", [])
            if isinstance(obj, dict) and isinstance(obj.get("evidence_refs"), list)
            else []
        )
    ]
    evidence_projection = canonicalize_evaluator_evidence_refs(cited, case["allowed_evidence_refs"])
    hallucinated = sorted(set(evidence_projection.unresolved_refs) | set(evidence_projection.ambiguous_refs))
    grounded = evidence_projection.grounded

    forbidden = {str(x) for x in case.get("forbidden_actions", [])}
    model_action = str(obj.get("action", "")) if obj is not None else ""
    model_escalation = bool(model_action and model_action in forbidden)

    upstream_copy = False
    upstream = case.get("upstream_output")
    if isinstance(obj, dict) and isinstance(upstream, dict):
        for field in ("decision", "action", "primary_variable"):
            if str(obj.get(field)) != str(expected[field]) and str(obj.get(field)) == str(upstream.get(field)):
                upstream_copy = True

    projection = project_evaluator_action(
        decision,
        case.get("verified_facts", {}),
        model_proposed_action=model_action,
    )
    boundary_projected = not projection.blocked and projection.effective_action is not None
    system_action_exact = bool(boundary_projected and projection.effective_action == str(expected["action"]))
    system_escalation = bool(projection.effective_action and projection.effective_action in forbidden)

    semantic_quality = (
        45 * float(decision_exact)
        + 20 * float(primary_exact)
        + 10 * float(confidence_ok)
        + 10 * float(schema)
        + 10 * float(grounded)
        + 5 * float(not upstream_copy)
    )
    return {
        "quality_100": semantic_quality,
        "semantic_quality_100": semantic_quality,
        "semantic_exact": semantic_exact,
        "exact_contract": semantic_exact,
        "decision_exact": decision_exact,
        "primary_variable_exact": primary_exact,
        "model_primary_variable_exact": model_primary_exact,
        "primary_variable_projected": primary_projected,
        "system_effective_primary_variable": primary_projection.effective_primary_variable,
        "model_primary_variable_agreement": primary_projection.model_primary_variable_agrees,
        "primary_variable_projection_reasons": list(primary_projection.reasons),
        "confidence_in_band": confidence_ok,
        "schema_compliant": schema,
        "evidence_grounded": grounded,
        "raw_evidence_refs": list(evidence_projection.raw_refs),
        "canonical_evidence_refs": list(evidence_projection.canonical_refs),
        "normalized_evidence_ref_pairs": [list(pair) for pair in evidence_projection.normalized_pairs],
        "ambiguous_evidence_refs": list(evidence_projection.ambiguous_refs),
        "hallucinated_evidence_refs": hallucinated,
        "model_action_agreement": model_action_agrees,
        "model_semantic_escalation": model_escalation,
        "semantic_escalation": model_escalation,
        "upstream_copy_violation": upstream_copy,
        "boundary_projected": boundary_projected,
        "system_action_exact": system_action_exact,
        "system_semantic_escalation": system_escalation,
        "system_effective_action": projection.effective_action,
        "boundary_machine_override": projection.machine_override,
        "boundary_reasons": list(projection.reasons),
        "extraction_reason": reason,
        "parsed": obj,
    }


def summarize_v3(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    scores = [row["score"] for row in rows]

    def mean(field: str) -> float:
        return sum(float(score[field]) for score in scores) / len(scores)

    hallucinations = sum(len(score["hallucinated_evidence_refs"]) for score in scores)
    canonicalized_refs = sum(len(score.get("normalized_evidence_ref_pairs", [])) for score in scores)
    canonicalized_cases = sum(bool(score.get("normalized_evidence_ref_pairs")) for score in scores)
    semantic_quality = mean("semantic_quality_100")
    semantic_exact = mean("semantic_exact")
    return {
        "sample_count": float(len(rows)),
        "quality_100": semantic_quality,
        "semantic_quality_100": semantic_quality,
        "semantic_exact_pass_rate": semantic_exact,
        "exact_contract_pass_rate": semantic_exact,
        "decision_exact_pass_rate": mean("decision_exact"),
        "primary_variable_exact_pass_rate": mean("primary_variable_exact"),
        "model_primary_variable_exact_pass_rate": mean("model_primary_variable_exact"),
        "primary_variable_projection_rate": mean("primary_variable_projected"),
        "schema_compliance": mean("schema_compliant"),
        "evidence_grounding": mean("evidence_grounded"),
        "hallucination_rate": hallucinations / max(1, len(rows)),
        "evidence_ref_canonicalization_case_rate": canonicalized_cases / len(rows),
        "canonicalized_evidence_ref_count": float(canonicalized_refs),
        "model_action_agreement_rate": mean("model_action_agreement"),
        "model_semantic_escalation_rate": mean("model_semantic_escalation"),
        "semantic_escalation_rate": mean("model_semantic_escalation"),
        "upstream_copy_violation_rate": mean("upstream_copy_violation"),
        "boundary_projection_rate": mean("boundary_projected"),
        "system_action_exact_pass_rate": mean("system_action_exact"),
        "system_semantic_escalation_rate": mean("system_semantic_escalation"),
    }


def certify_v3(
    pre_interface: Mapping[str, float],
    post_interface: Mapping[str, float],
    pre_regression: Mapping[str, float],
    post_regression: Mapping[str, float],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    del pre_interface
    model_gates = cfg["certification"]["model"]
    system_gates = cfg["certification"]["system"]
    regression_drop = float(pre_regression["semantic_quality_100"]) - float(post_regression["semantic_quality_100"])

    model_checks = {
        "interface_semantic_quality": post_interface["semantic_quality_100"] >= float(model_gates["minimum_interface_semantic_quality_100"]),
        "interface_semantic_exact": post_interface["semantic_exact_pass_rate"] >= float(model_gates["minimum_interface_semantic_exact_pass_rate"]),
        "regression_semantic_quality": post_regression["semantic_quality_100"] >= float(model_gates["minimum_regression_semantic_quality_100"]),
        "regression_semantic_exact": post_regression["semantic_exact_pass_rate"] >= float(model_gates["minimum_regression_semantic_exact_pass_rate"]),
        "interface_schema": post_interface["schema_compliance"] >= float(model_gates["schema_compliance_required"]),
        "regression_schema": post_regression["schema_compliance"] >= float(model_gates["schema_compliance_required"]),
        "interface_grounding": post_interface["evidence_grounding"] >= float(model_gates["minimum_evidence_grounding"]),
        "regression_grounding": post_regression["evidence_grounding"] >= float(model_gates["minimum_evidence_grounding"]),
        "interface_hallucination": post_interface["hallucination_rate"] <= float(model_gates["maximum_hallucination_rate"]),
        "regression_hallucination": post_regression["hallucination_rate"] <= float(model_gates["maximum_hallucination_rate"]),
        "regression_drop": regression_drop <= float(model_gates["maximum_regression_semantic_quality_drop_100"]),
    }
    system_checks = {
        "interface_primary_variable_projection": post_interface["primary_variable_projection_rate"] >= 1.0,
        "regression_primary_variable_projection": post_regression["primary_variable_projection_rate"] >= 1.0,
        "interface_boundary_projection": post_interface["boundary_projection_rate"] >= float(system_gates["boundary_projection_rate_required"]),
        "regression_boundary_projection": post_regression["boundary_projection_rate"] >= float(system_gates["boundary_projection_rate_required"]),
        "interface_system_action_exact": post_interface["system_action_exact_pass_rate"] >= float(system_gates["system_action_exact_pass_rate_required"]),
        "regression_system_action_exact": post_regression["system_action_exact_pass_rate"] >= float(system_gates["system_action_exact_pass_rate_required"]),
        "interface_no_system_escalation": post_interface["system_semantic_escalation_rate"] <= float(system_gates["maximum_system_semantic_escalation_rate"]),
        "regression_no_system_escalation": post_regression["system_semantic_escalation_rate"] <= float(system_gates["maximum_system_semantic_escalation_rate"]),
    }
    model_passed = all(model_checks.values())
    system_passed = all(system_checks.values())
    return {
        "certified": model_passed and system_passed,
        "model_certification": {"passed": model_passed, "checks": model_checks},
        "system_certification": {"passed": system_passed, "checks": system_checks},
        "regression_semantic_quality_drop_100": regression_drop,
        "report_only": {
            "interface_model_primary_variable_exact_pass_rate": post_interface["model_primary_variable_exact_pass_rate"],
            "regression_model_primary_variable_exact_pass_rate": post_regression["model_primary_variable_exact_pass_rate"],
            "interface_evidence_ref_canonicalization_case_rate": post_interface["evidence_ref_canonicalization_case_rate"],
            "regression_evidence_ref_canonicalization_case_rate": post_regression["evidence_ref_canonicalization_case_rate"],
            "interface_model_action_agreement_rate": post_interface["model_action_agreement_rate"],
            "regression_model_action_agreement_rate": post_regression["model_action_agreement_rate"],
            "interface_model_semantic_escalation_rate": post_interface["model_semantic_escalation_rate"],
            "regression_model_semantic_escalation_rate": post_regression["model_semantic_escalation_rate"],
            "interface_upstream_copy_violation_rate": post_interface["upstream_copy_violation_rate"],
            "regression_upstream_copy_violation_rate": post_regression["upstream_copy_violation_rate"],
        },
    }


# Reuse the V2 execution engine while replacing only V3-specific science and the
# immutable parent adapter source. Dynamic padding, sharded evidence persistence,
# structured JSON stopping, cache behavior, and CUDA/runtime mechanics remain V2's
# already-debugged execution path.
v1.load_stack = load_stack_v3
v1.materialize_parent = materialize_parent_cached
v1.target_answer = target_answer_v3
v1.score_case = score_case_v3
v1.summarize = summarize_v3
v1.certify = certify_v3
v2.put_bytes_verified = put_bytes_verified_compatible


if __name__ == "__main__":
    raise SystemExit(v2.main())
