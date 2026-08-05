from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest

from hephaestus.config_loader import ConfigError
from hephaestus.evaluation import ExperimentEvaluationService
from hephaestus.interfaces.services import ExperimentEvaluationService as ExperimentEvaluationProtocol
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingRunHandle
from hephaestus.scoring.behavioral import evaluate_behavioral_sample


BASELINE_OUTPUTS = {
    "instruction_triplet": "alpha beta.",
    "planet_fact": "Mars exists.",
    "observatory_continuation": "People waited.",
    "structured_planet_answer": "Mars, probably.",
    "anti_repetition": "Evidence helps.",
    "brief_termination": "Rollback helps.",
}

GOOD_OUTPUTS = {
    "instruction_triplet": "alpha beta gamma.",
    "planet_fact": "Mars is commonly called the Red Planet.",
    "observatory_continuation": "The astronomer found a lantern and checked the silent instruments.",
    "structured_planet_answer": '{"answer":"Mars","confidence":0.9}',
    "anti_repetition": "Repeated evidence raises confidence by showing that an effect is consistent.",
    "brief_termination": "Rollback safety restores a known good state after failure.",
}


class FavorableJudge:
    def score(self, task: Mapping[str, object], response: str) -> Mapping[str, float]:
        return {"instruction_adherence": 0.95, "relevance": 0.95, "coherence": 0.95}


def _proposal() -> ExperimentProposal:
    return ExperimentProposal(
        experiment_id="exp-semantic",
        run_id="candidate-1",
        lineage_id="lineage-main",
        stage_name="stabilization",
        diagnosis_report_id="diag-1",
        intervention_id="intervention-1",
        primary_variable="training_recipe",
        baseline_ref="baseline-1",
    )


def _run(
    service: ExperimentEvaluationService,
    run_id: str,
    outputs: dict[str, str],
    *,
    experiment_id: str = "exp-semantic",
    decoding_config: dict[str, object] | None = None,
    content_hash: str | None = None,
    eval_pack_id: str | None = None,
    integrity_level: str = "content_hash_verified",
    omit: tuple[str, int] | None = None,
) -> TrainingRunHandle:
    seeds = list(service.pack["eval_pack"]["decoding_config"]["seeds"])
    samples = [
        {
            "task_id": task_id,
            "seed": seed,
            "output": output,
            "evidence_ref": f"artifacts/{run_id}/{task_id}-{seed}.json",
        }
        for task_id, output in outputs.items()
        for seed in seeds
        if omit != (task_id, seed)
    ]
    return TrainingRunHandle(
        run_id=run_id,
        experiment_id=experiment_id,
        backend_id="fixture",
        status="completed",
        metadata={
            "semantic_evaluation": {
                "eval_pack_id": eval_pack_id if eval_pack_id is not None else service.pack["eval_pack_id"],
                "eval_pack_version": service.pack["eval_pack_version"],
                "integrity_level": integrity_level,
                "content_hash": content_hash if content_hash is not None else service.pack["content_hash"],
                "decoding_config": decoding_config or service.pack["eval_pack"]["decoding_config"],
                "report_ref": f"reports/eval-{run_id}.json",
                "evidence_refs": [f"artifacts/{run_id}/semantic-evidence.json"],
                "samples": samples,
            }
        },
    )


def _comparison(
    service: ExperimentEvaluationService,
    candidate_outputs: dict[str, str] | None = None,
):
    outputs = candidate_outputs or GOOD_OUTPUTS
    return service.compare(
        _proposal(),
        [
            _run(service, "baseline-1", BASELINE_OUTPUTS, experiment_id="prior-experiment"),
            _run(service, "candidate-1", outputs),
            _run(service, "candidate-2", outputs),
        ],
    )


def test_baseline_and_candidate_fixture_produce_valid_improved_comparison() -> None:
    service = ExperimentEvaluationService()
    comparison = _comparison(service)

    assert isinstance(service, ExperimentEvaluationProtocol)
    assert comparison.primary_outcome == "improved"
    assert comparison.deterministic_gate_status == "passed"
    assert comparison.baseline_run_id == "baseline-1"
    assert comparison.candidate_run_ids == ["candidate-1", "candidate-2"]
    assert len(comparison.evaluation_report_refs) == 3
    assert comparison.effect_summary["overall_delta"] >= service.minimum_practical_improvement
    assert comparison.metadata["does_not_promote"] is True
    assert comparison.metadata["human_review_bundle"]["samples"]
    review_row = comparison.metadata["human_review_bundle"]["samples"][0]
    assert "baseline_output_ref" in review_row and "candidate_output_ref" in review_row
    assert "baseline_output" not in review_row and "candidate_output" not in review_row


def test_hard_repetition_and_termination_regressions_block_improved() -> None:
    service = ExperimentEvaluationService(judge_adapter=FavorableJudge())
    regressed = dict(GOOD_OUTPUTS)
    regressed["anti_repetition"] = "evidence confidence evidence confidence evidence confidence evidence confidence"
    regressed["brief_termination"] = "Rollback safety matters because"

    comparison = _comparison(service, regressed)

    assert comparison.primary_outcome == "regressed"
    assert comparison.deterministic_gate_status == "failed"
    failures = comparison.effect_summary["deterministic"]["candidate_hard_failures"]
    assert any("anti_loop" in item for item in failures)
    assert any("brief_termination" in item for item in failures)
    assert comparison.effect_summary["judge"]["disagreements"]
    assert comparison.effect_summary["judge"]["deterministic_precedence"] is True


def test_mixed_dimension_changes_do_not_become_improved() -> None:
    service = ExperimentEvaluationService()
    baseline = dict(GOOD_OUTPUTS)
    baseline["instruction_triplet"] = BASELINE_OUTPUTS["instruction_triplet"]
    mixed = dict(GOOD_OUTPUTS)
    mixed["planet_fact"] = "A distant world exists."
    mixed["observatory_continuation"] = "The team checked the instruments and waited."
    mixed["anti_repetition"] = "Repeated trials can help."
    comparison = service.compare(
        _proposal(),
        [
            _run(service, "baseline-1", baseline, experiment_id="prior-experiment"),
            _run(service, "candidate-1", mixed),
            _run(service, "candidate-2", mixed),
        ],
    )

    assert comparison.primary_outcome in {"mixed", "inconclusive"}
    effects = {item["effect"] for item in comparison.effect_summary["dimensions"].values()}
    assert "improved" in effects and "regressed" in effects


def test_repeated_results_change_variance_risk_and_confidence() -> None:
    service = ExperimentEvaluationService()
    consistent = _comparison(service)
    variable = dict(GOOD_OUTPUTS)
    variable["planet_fact"] = "A distant world exists."
    variable["observatory_continuation"] = "The team waited quietly."
    variable["anti_repetition"] = "Repeated trials can help."
    variable_comparison = service.compare(
        _proposal(),
        [
            _run(service, "baseline-1", BASELINE_OUTPUTS, experiment_id="prior-experiment"),
            _run(service, "candidate-1", GOOD_OUTPUTS),
            _run(service, "candidate-2", variable),
        ],
    )

    assert consistent.variance_risk == "low"
    assert variable_comparison.variance_risk in {"moderate", "high"}
    assert variable_comparison.confidence < consistent.confidence


def test_decoding_or_pack_mismatch_invalidates_comparison() -> None:
    service = ExperimentEvaluationService()
    mismatched_decoding = dict(service.pack["eval_pack"]["decoding_config"])
    mismatched_decoding["temperature"] = 0.7
    comparison = service.compare(
        _proposal(),
        [
            _run(service, "baseline-1", BASELINE_OUTPUTS, experiment_id="prior-experiment"),
            _run(service, "candidate-1", GOOD_OUTPUTS, decoding_config=mismatched_decoding),
        ],
    )

    assert comparison.primary_outcome == "invalid_comparison"
    assert comparison.deterministic_gate_status == "incompatible"
    assert any(issue.code == "decoding_settings_mismatch" and issue.blocking for issue in comparison.issues)

    missing_identity = service.compare(
        _proposal(),
        [
            _run(service, "baseline-1", BASELINE_OUTPUTS, experiment_id="prior-experiment"),
            _run(service, "candidate-1", GOOD_OUTPUTS, eval_pack_id=""),
        ],
    )
    assert missing_identity.primary_outcome == "invalid_comparison"
    assert any(issue.code == "eval_pack_identity_missing" for issue in missing_identity.issues)

    bad_hash = service.compare(
        _proposal(),
        [
            _run(service, "baseline-1", BASELINE_OUTPUTS, experiment_id="prior-experiment"),
            _run(service, "candidate-1", GOOD_OUTPUTS, content_hash="not-the-pack-hash"),
        ],
    )
    assert bad_hash.primary_outcome == "invalid_comparison"
    assert any(issue.code == "eval_pack_content_hash_mismatch" for issue in bad_hash.issues)


def test_missing_required_sample_is_inconclusive_and_lowers_confidence() -> None:
    service = ExperimentEvaluationService()
    comparison = service.compare(
        _proposal(),
        [
            _run(service, "baseline-1", BASELINE_OUTPUTS, experiment_id="prior-experiment"),
            _run(service, "candidate-1", GOOD_OUTPUTS, omit=("planet_fact", 11)),
            _run(service, "candidate-2", GOOD_OUTPUTS),
        ],
    )

    assert comparison.primary_outcome == "inconclusive"
    assert comparison.deterministic_gate_status == "incomplete"
    assert comparison.confidence <= 0.45
    assert any(issue.code == "required_samples_missing" for issue in comparison.issues)


def test_pack_is_versioned_frozen_and_hash_verified() -> None:
    service = ExperimentEvaluationService()
    pack = service.pack

    assert pack["eval_pack_id"] == "semantic_behavior"
    assert pack["eval_pack_version"] == "1.0.0"
    assert pack["frozen"] is True
    assert pack["content_hash_verified"] is True
    assert pack["mutation_policy"] == "new_version_required"
    assert pack["task_bundles"]
    assert pack["success_semantics"] and pack["failure_semantics"]
    assert pack["stage_applicability"] and pack["expected_evidence"]


def test_pack_loader_rejects_false_content_hash_claim(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree("configs", config_dir)
    pack_path = config_dir / "eval_packs" / "semantic_behavior_v1.yaml"
    payload = json.loads(pack_path.read_text())
    payload["content_hash"] = "0" * 64
    pack_path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(ConfigError, match="content hash mismatch"):
        ExperimentEvaluationService(config_dir=config_dir)


def test_behavioral_checks_detect_malformed_structure_without_network() -> None:
    service = ExperimentEvaluationService()
    task = service._tasks()["structured_planet_answer"]
    malformed = evaluate_behavioral_sample(task, "not json", seed=11)
    valid = evaluate_behavioral_sample(task, '{"answer":"Mars","confidence":0.8}', seed=11)

    assert malformed.deterministic_passed is False
    assert "planet_json" in malformed.failed_hard_checks
    assert valid.deterministic_passed is True
