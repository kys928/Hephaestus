"""Inert, source-pinned D metrics -> genuine deterministic Diagnosis and Planner.

This consumes a *historical eight-state development observation*, not the
frozen certified production Evaluator, and never infers a repair hypothesis
from poor task accuracy alone. No provider, model inference or training.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

from hephaestus.diagnosis.pinned_historical import _parse_json, PinnedEvidenceError
from hephaestus.diagnosis.service import EvidenceBasedDiagnosisService
from hephaestus.planning.service import ClosedLoopExperimentPlanner
from hephaestus.schemas.diagnosis_contract import DiagnosisRequest

SHA_REPORT = "sha256:7398eeed835422151678e4e1a79ee4c0d8ead4daf9dba61dd4333aa9acfbf953"
SHA_ARCHIVE = "sha256:25ff27dea3153f601876d3ffb6bf5628cba734596f89364e7911286a1e4173ff"
D_SOURCE_SHA = "ffa0458d5850f9d572a06d5ee6660fd63fc8acc6"
D_ADAPTER_SHA = "5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1"
ONTOLOGY = "historical-semantic-invariance-evaluator.v1"
SHARD_ROOT = ("hephaestus/scientific/v5/evaluator_semantic_repair_v2/"
              "evaluator-semantic-v2-D-38945578442/checkpoints/step-096/evaluation/samples/")
SHARDS = (
    {"key": SHARD_ROOT + "shard-0001.json", "bytes": 124152,
     "sha256": "934363b71fd76d835b1d05bbc8892645605b3644e1415fa3a82d82e385f67dbc"},
    {"key": SHARD_ROOT + "shard-0002.json", "bytes": 124363,
     "sha256": "471897bcfaa66a707599f234e5a7405b8d3783c67d876f34ff029f3c8fc2d1e1"},
)
BY_STATE_CORRECT = {
    "certification_ready": 7, "equivalent": 0, "improved": 4,
    "incomplete_evidence": 5, "inconclusive": 8,
    "recheck_required": 0, "regressed": 4, "scientific_rejection": 5,
}
REQUIRED_KEYS = frozenset({
    "schema", "source_workflow_run_id", "source_artifact_id",
    "original_aggregate_report_sha256", "original_archive_zip_sha256",
    "historical_source_repo_sha", "historical_adapter_sha256_claim",
    "ontology", "partition", "sample_count", "strict_correct_count",
    "historical_saved_correct_count", "strict_disagreements_with_saved",
    "by_state_correct", "by_state_total", "source_shards",
    "s3_bytes_sha256_verified", "model_weights_independently_verified",
    "scientific_causality_established", "independent_final_assessment",
    "protected_final_holdout_accessed", "scientific_certification",
    "no_model_inference_or_training_performed",
})


def validate_aggregate(value: object) -> dict:
    """Accept only the independently recovered, documented aggregate identity.

    A locally authored summary alone cannot verify original model weights or
    establish causal training failures. Do not import any raw asserted signals.
    """
    if not isinstance(value, dict) or set(value) != REQUIRED_KEYS:
        raise PinnedEvidenceError("unrecognized or extended D observation manifest")
    expected = {
        "schema": "prometheus-granite-d-observation.v1",
        "source_workflow_run_id": 41609821443,
        "source_artifact_id": 13217803440,
        "original_aggregate_report_sha256": SHA_REPORT,
        "original_archive_zip_sha256": SHA_ARCHIVE,
        "historical_source_repo_sha": D_SOURCE_SHA,
        "historical_adapter_sha256_claim": D_ADAPTER_SHA,
        "ontology": ONTOLOGY,
        "partition": "exposed_development",
        "sample_count": 64,
        "strict_correct_count": 33,
        "historical_saved_correct_count": 33,
        "strict_disagreements_with_saved": 0,
        "s3_bytes_sha256_verified": True,
        "model_weights_independently_verified": False,
        "scientific_causality_established": False,
        "independent_final_assessment": False,
        "protected_final_holdout_accessed": False,
        "scientific_certification": False,
        "no_model_inference_or_training_performed": True,
    }
    for key, intended in expected.items():
        if type(value[key]) is not type(intended) or value[key] != intended:
            raise PinnedEvidenceError(f"historical D observation identity or claim drift: {key}")
    if value["by_state_correct"] != BY_STATE_CORRECT:
        raise PinnedEvidenceError("historical D by-state measurements differ from recomputed report")
    if value["by_state_total"] != {key: 8 for key in BY_STATE_CORRECT}:
        raise PinnedEvidenceError("unbalanced or altered D development screen")
    if sum(value["by_state_correct"].values()) != 33 or sum(value["by_state_total"].values()) != 64:
        raise PinnedEvidenceError("historical D aggregate count inconsistent")
    if value["source_shards"] != list(SHARDS):
        raise PinnedEvidenceError("D source shard identities are not those independently checked")
    return value


def preflight(aggregate: object) -> dict:
    """Call Hephaestus's existing real deterministic services, not an LLM stand-in."""
    verified = validate_aggregate(aggregate)
    # This is *not* an eval_report with an independently verified frozen eval
    # pack / scorecard and model lineage. It must never forge those signals.
    observation = {
        "evidence_kind": "historical_observation",
        "source_ref": SHA_REPORT,
        "summary": (
            "Prior exposed eight-state Granite D development decisions: "
            "33 of 64 correct; equivalent 0 of 8; recheck_required 0 of 8. "
            "Raw-output S3 hashes and strict scoring reproduced; model weights "
            "and independent final evaluation remain unverified."
        ),
        "confidence": 0.8,  # confidence in the scoped observation, not a cause
    }
    request = DiagnosisRequest(
        request_id="granite-d-verified-historical-observation",
        run_id="granite-d-historical-readonly",
        lineage_id="historical-D-not-production-certified-parent",
        stage_name="continuation_repair",
        observed_failures=[observation],
        evidence_refs=[],
        requested_by="read_only_evidence_preflight",
    )
    diagnosis = EvidenceBasedDiagnosisService().diagnose(request)
    proposals = list(ClosedLoopExperimentPlanner().propose_interventions(diagnosis))
    if not proposals or any(p.intervention_kind not in {"collect_more_evidence", "repair_evaluation"}
                            for p in proposals):
        raise PinnedEvidenceError("unverified historical results must not admit a training intervention")
    return {
        "schema": "prometheus-granite-d-diagnosis-preflight.v1",
        "source_aggregate_report_sha256": verified["original_aggregate_report_sha256"],
        "source_ontology": ONTOLOGY,
        "historical_evidence_exposed": True,
        "source_is_not_certified_parent": True,
        "decision_owner": "HEPHAESTUS_DETERMINISTIC_POLICY",
        "model_backed_specialist_inference": False,
        "training_executed": False,
        "checkpoint_promoted": False,
        "scientific_repair_established": False,
        "independent_certification": False,
        "diagnosis": diagnosis.to_dict(),
        "interventions": [p.to_dict() for p in proposals],
        "next_scientific_gate": "authenticate parent model, frozen oracle, and source training/evaluation contract",
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.aggregate.read_bytes()
    if len(raw) > 32_768:
        raise PinnedEvidenceError("observation manifest byte bound exceeded")
    parsed = _parse_json(raw.decode("utf-8"))
    report = preflight(parsed)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("GRANITE_D_DIAGNOSIS_PREFLIGHT", json.dumps({
        "status": report["diagnosis"]["status"],
        "intervention_kinds": [p["intervention_kind"] for p in report["interventions"]],
        "model_backed_specialist_inference": False, "training_executed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
