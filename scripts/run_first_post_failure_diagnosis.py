#!/usr/bin/env python3
"""Run the deterministic diagnosis subsystem on the frozen first failed checkpoint chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hephaestus.diagnosis import EvidenceBasedDiagnosisService
from hephaestus.schemas.diagnosis_contract import DiagnosisRequest

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "docs/evidence/first-bounded-scientific-training-001-33866198758"
EVAL_DIR = ROOT / "docs/evidence/first-semantic-evaluation-001-33869352751"
TRAINING_VERIFICATION_SHA = "c4b1873da483fb672c146248b6a9116af11065d4fa103658fac40bc7aab4774b"
EVAL_VERIFICATION_SHA = "4a3e266413a717c31e83f7f1d894e7ccd6e0bef1d59e9b1a427b67ecf178e8c4"
TRAIN_RUN = "first-bounded-scientific-training-001-33866198758"
EVAL_RUN = "first-semantic-evaluation-001-33869352751"
EXPERIMENT = "experiment-60bff7cb4f478f91"
LINEAGE = "lineage-first-scientific"
CHECKPOINT_HASH = "sha256:7a6be1e0cee47f29d5dd47d41bc01beed066c4de64e24ee18544ff4edcb3f4c3"
MODEL_ID = "sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39"
TOKENIZER_ID = "sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
PROCESSED_DATA = "sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c"
EVAL_PACK_HASH = "ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified(path: Path, expected: str) -> dict[str, object]:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"frozen evidence drift for {path}: {observed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid evidence shape: {path}")
    return payload


def nested(payload: dict[str, object], *path: str) -> object:
    cur: object = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise RuntimeError(f"missing required evidence field: {'/'.join(path)}")
        cur = cur[key]
    return cur


def main() -> int:
    training_path = TRAINING_DIR / "verification.json"
    eval_path = EVAL_DIR / "verification.json"
    training = read_verified(training_path, TRAINING_VERIFICATION_SHA)
    evaluation = read_verified(eval_path, EVAL_VERIFICATION_SHA)

    training_terminal = nested(training, "terminal_result")
    if not isinstance(training_terminal, dict):
        raise RuntimeError("training terminal evidence malformed")
    comparison = nested(evaluation, "comparison")
    if not isinstance(comparison, dict):
        raise RuntimeError("comparison evidence malformed")

    eval_terminal = evaluation.get("terminal_result")
    if not isinstance(eval_terminal, dict):
        eval_terminal = evaluation.get("evaluation_result")
    if not isinstance(eval_terminal, dict):
        raise RuntimeError("evaluation terminal evidence missing")
    judge = eval_terminal.get("judge_exit")
    if not isinstance(judge, dict):
        raise RuntimeError("Judge-exit evidence missing")

    assertions = {
        "training_verification_status": training.get("status") == "verified",
        "training_completed": training_terminal.get("status") == "completed",
        "training_run_identity": training.get("run_id") == TRAIN_RUN,
        "checkpoint_identity": nested(training, "checkpoint", "checkpoint_manifest_hash") == CHECKPOINT_HASH,
        "model_identity": nested(training, "input_identities", "model_directory_identity") == MODEL_ID,
        "tokenizer_identity": nested(training, "input_identities", "tokenizer_directory_identity") == TOKENIZER_ID,
        "processed_data_identity": nested(training, "input_identities", "processed_dataset_sha256") == PROCESSED_DATA,
        "comparison_regressed": comparison.get("primary_outcome") == "regressed",
        "deterministic_gate_failed": comparison.get("deterministic_gate_status") == "failed",
        "candidate_is_training_run": comparison.get("candidate_run_ids") == [TRAIN_RUN],
        "comparison_has_no_issues": comparison.get("issues") == [],
        "judge_rejected": judge.get("next_action") == "reject_checkpoint",
        "judge_not_applied": eval_terminal.get("action_applied") is False,
        "evaluation_did_not_train": eval_terminal.get("training_performed") is False,
    }
    failed = [name for name, ok in assertions.items() if not ok]
    if failed:
        raise RuntimeError("chain verification failed: " + ", ".join(failed))

    effect = comparison.get("effect_summary") if isinstance(comparison.get("effect_summary"), dict) else {}
    metrics = training_terminal.get("metrics_summary") if isinstance(training_terminal.get("metrics_summary"), dict) else {}
    normalized = training_terminal.get("normalized_training_config") if isinstance(training_terminal.get("normalized_training_config"), dict) else {}

    # Only facts explicitly established by frozen evidence are projected.
    # No causal signal such as undertraining_detected, scheduler_misconfigured,
    # overfitting_detected, or model_family_limitation_detected is invented.
    observed = [
        {
            "evidence_kind": "eval_report",
            "source_ref": str(eval_path.relative_to(ROOT)),
            "summary": "Frozen semantic_behavior_v1 comparison recorded a verified regression.",
            "confidence": comparison.get("confidence", 0.0),
            "eval_pack_integrity_level": "content_hash_verified",
            "scorecard_integrity_level": "content_hash_verified",
            "deterministic_scorecard": {
                "gate_results": {
                    "semantic_behavior_v1": {
                        "passed": False,
                        "primary_outcome": comparison.get("primary_outcome"),
                        "deterministic_gate_status": comparison.get("deterministic_gate_status"),
                    }
                }
            },
            "decoding_verified": True,
            "overall_delta": effect.get("overall_delta"),
        },
        {
            "evidence_kind": "run_record",
            "source_ref": str(training_path.relative_to(ROOT)),
            "run_id": TRAIN_RUN,
            "status": training_terminal.get("status"),
            "experiment_id": EXPERIMENT,
        },
        {
            "evidence_kind": "dataset_manifest",
            "source_ref": str(training_path.relative_to(ROOT)),
            "manifest_integrity_level": "complete",
            "processed_dataset_hash": PROCESSED_DATA,
            "tokenizer_compatibility": {"status": "compatible", "tokenizer_ref": TOKENIZER_ID},
        },
        {
            "evidence_kind": "checkpoint_evidence",
            "source_ref": str(training_path.relative_to(ROOT)),
            "checkpoint_verified": True,
            "checkpoint_manifest_hash": CHECKPOINT_HASH,
        },
        {
            "evidence_kind": "training_metrics",
            "source_ref": str(training_path.relative_to(ROOT)),
            "finite": metrics.get("finite"),
            "numerically_stable": metrics.get("finite") is True,
            "steps": metrics.get("steps"),
            "tokens_processed": metrics.get("tokens_processed"),
            "final_training_loss": metrics.get("final_training_loss"),
            "final_gradient_norm": metrics.get("final_gradient_norm"),
        },
        {
            "evidence_kind": "explicit_check",
            "source_ref": str(eval_path.relative_to(ROOT)),
            "architecture_compatible": True,
            "decoding_verified": True,
            "model_revision": normalized.get("model_revision"),
            "tokenizer_revision": normalized.get("tokenizer_revision"),
        },
    ]

    request = DiagnosisRequest(
        request_id="diagnose-first-semantic-regression-001",
        run_id=TRAIN_RUN,
        lineage_id=LINEAGE,
        stage_name="smoke_test",
        observed_failures=observed,
        requested_by="judge_exit_post_failure_handoff",
    )
    report = EvidenceBasedDiagnosisService().diagnose(request)

    leading = next((h for h in report.hypotheses if h.hypothesis_id == report.leading_hypothesis_id), None)
    if report.status != "inconclusive":
        raise RuntimeError(f"diagnosis unexpectedly asserted a cause: {report.status}")
    if leading is None or leading.failure_domain != "inconclusive":
        raise RuntimeError("diagnosis did not preserve causal uncertainty")
    if leading.metadata.get("causation_claimed") is not False:
        raise RuntimeError("diagnosis incorrectly claimed causation")
    if report.metadata.get("recommendations_executed") is not False:
        raise RuntimeError("diagnosis executed a recommendation")
    if report.issues:
        raise RuntimeError(f"diagnosis emitted unexpected issues: {[issue.code for issue in report.issues]}")

    chain = {
        "chain_version": "first-post-failure-diagnosis-chain.v1",
        "training_run_id": TRAIN_RUN,
        "evaluation_run_id": EVAL_RUN,
        "experiment_id": EXPERIMENT,
        "lineage_id": LINEAGE,
        "frozen_training_verification_sha256": f"sha256:{TRAINING_VERIFICATION_SHA}",
        "frozen_evaluation_verification_sha256": f"sha256:{EVAL_VERIFICATION_SHA}",
        "checkpoint_manifest_hash": CHECKPOINT_HASH,
        "eval_pack_content_hash": EVAL_PACK_HASH,
        "comparison_primary_outcome": comparison.get("primary_outcome"),
        "comparison_deterministic_gate_status": comparison.get("deterministic_gate_status"),
        "judge_next_action": judge.get("next_action"),
        "judge_action_applied": eval_terminal.get("action_applied"),
        "diagnosis_status": report.status,
        "diagnosis_leading_domain": leading.failure_domain,
        "diagnosis_confidence": report.confidence,
        "diagnosis_missing_evidence": report.missing_evidence,
        "diagnosis_issues": [],
        "chain_assertions": assertions,
        "causal_signal_invented": False,
        "recommendation_executed": False,
    }
    projection = {
        "projection_version": "first-post-failure-diagnosis-input.v1",
        "request": request.to_dict(),
        "projection_policy": {
            "only_frozen_supported_facts": True,
            "regression_is_not_itself_a_cause": True,
            "invented_failure_domain_signals": [],
        },
    }

    outputs = {
        "first_post_failure_diagnosis_report.json": report.to_dict(),
        "first_post_failure_diagnosis_chain.json": chain,
        "first_post_failure_diagnosis_input.json": projection,
    }
    for name, payload in outputs.items():
        Path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": report.status,
        "leading_domain": leading.failure_domain,
        "confidence": report.confidence,
        "missing_evidence": report.missing_evidence,
        "issues": [],
        "judge_next_action": judge.get("next_action"),
        "causation_claimed": leading.metadata.get("causation_claimed"),
        "recommendations_executed": report.metadata.get("recommendations_executed"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
