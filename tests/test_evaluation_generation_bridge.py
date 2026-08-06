from __future__ import annotations

import json
from pathlib import Path

from hephaestus.evaluation import ExperimentEvaluationService
from hephaestus.generation import (
    DeterministicFakeGenerationBackend,
    EvaluationGenerationService,
    StagedExperimentEvaluationAdapter,
    StagedGenerationAdapter,
    TransformersCausalLMGenerationBackend,
)
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingRunHandle
from hephaestus.control.staged_state import StagedOperationRequest


def _output(_run_id: str, task) -> str:
    return {
        "instruction_triplet": "alpha beta gamma.",
        "planet_fact": "Mars is commonly called the Red Planet.",
        "observatory_continuation": "They lit a lantern and continued observing after the storm.",
        "structured_planet_answer": '{"answer":"Mars","confidence":0.99}',
        "anti_repetition": "Repeated evidence increases confidence by showing that results remain consistent.",
        "brief_termination": "Rollback safety restores a stable state after failure.",
    }[task.task_id]


def _handle(tmp_path: Path, run_id: str) -> tuple[TrainingRunHandle, str]:
    checkpoint = tmp_path / run_id / "checkpoint"
    checkpoint.mkdir(parents=True)
    handoff = checkpoint / "loading_instructions.json"
    handoff.write_text(
        json.dumps(
            {
                "backend": "transformers_causal_lm",
                "model_artifact_ref": str(checkpoint / "model"),
                "tokenizer_artifact_ref": str(checkpoint / "tokenizer"),
                "trust_remote_code": False,
            }
        ),
        encoding="utf-8",
    )
    return (
        TrainingRunHandle(
            run_id=run_id,
            experiment_id="exp-generation",
            backend_id="transformers_causal_lm",
            status="completed",
            checkpoint_refs=[str(checkpoint)],
            metadata={"generation_handoff_ref": str(handoff)},
        ),
        str(handoff),
    )


def _proposal() -> ExperimentProposal:
    return ExperimentProposal(
        experiment_id="exp-generation",
        run_id="candidate",
        lineage_id="lineage-1",
        stage_name="smoke_test",
        diagnosis_report_id="diagnosis-1",
        intervention_id="intervention-1",
        primary_variable="learning_rate",
        baseline_ref="baseline",
        status="approved",
    )


def _request(substep: str) -> StagedOperationRequest:
    return StagedOperationRequest(
        workflow_id="workflow-1",
        run_id="candidate",
        lineage_id="lineage-1",
        stage_name="smoke_test",
        phase="evaluator",
        substep=substep,
        operation_id=f"op-{substep}",
        attempt=1,
        input_refs=(),
        prior_outputs={},
    )


def test_frozen_pack_generation_hands_directly_to_evaluator(tmp_path: Path) -> None:
    baseline, baseline_handoff = _handle(tmp_path, "baseline")
    candidate, candidate_handoff = _handle(tmp_path, "candidate")
    backend = DeterministicFakeGenerationBackend(fallback=_output)
    service = EvaluationGenerationService(tmp_path / "artifacts", backend)

    plan = service.plan()
    assert plan.eval_pack_id == "semantic_behavior"
    assert plan.eval_pack_version == "1.0.0"
    assert plan.content_hash == "ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad"
    assert len(plan.tasks) == 18

    baseline_result = service.generate(
        baseline, generation_handoff_ref=baseline_handoff
    )
    candidate_result = service.generate(
        candidate, generation_handoff_ref=candidate_handoff
    )
    assert baseline_result.report.completed
    assert candidate_result.report.completed
    assert len(baseline_result.report.samples) == 18
    assert (
        baseline_result.report.generation_settings_id
        == candidate_result.report.generation_settings_id
    )

    comparison = ExperimentEvaluationService().compare(
        _proposal(), [baseline_result.run_handle, candidate_result.run_handle]
    )
    assert comparison.primary_outcome != "invalid_comparison"
    assert not [issue for issue in comparison.issues if issue.blocking]
    assert comparison.metadata["eval_pack_content_hash"] == plan.content_hash
    assert comparison.metadata["does_not_promote"] is True


def test_verified_samples_are_reused_and_corruption_is_regenerated(tmp_path: Path) -> None:
    run, handoff = _handle(tmp_path, "candidate")
    backend = DeterministicFakeGenerationBackend(fallback=_output)
    service = EvaluationGenerationService(tmp_path / "artifacts", backend)

    first = service.generate(run, generation_handoff_ref=handoff)
    second = service.generate(run, generation_handoff_ref=handoff)
    assert first.report.completed and second.report.completed
    assert backend.calls == 1
    assert [sample.sample_id for sample in first.report.samples] == [
        sample.sample_id for sample in second.report.samples
    ]

    sample_path = Path(first.report.samples[0].evidence_ref)
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    payload["output"] = "tampered"
    sample_path.write_text(json.dumps(payload), encoding="utf-8")

    repaired = service.generate(run, generation_handoff_ref=handoff)
    assert repaired.report.completed
    assert backend.calls == 2
    assert sample_path.with_suffix(".json.corrupt").exists()


def test_incomplete_run_and_missing_handoff_are_refused(tmp_path: Path) -> None:
    run = TrainingRunHandle(
        run_id="active",
        experiment_id="exp-generation",
        backend_id="transformers_causal_lm",
        status="running",
        checkpoint_refs=[],
    )
    result = EvaluationGenerationService(
        tmp_path / "artifacts", DeterministicFakeGenerationBackend(fallback=_output)
    ).generate(run)
    assert not result.report.completed
    assert {issue.code for issue in result.report.issues} == {
        "generation_run_not_completed",
        "generation_checkpoint_missing",
        "generation_handoff_missing",
    }
    assert "semantic_evaluation" not in result.run_handle.metadata


def test_staged_adapters_emit_required_records_and_evidence(tmp_path: Path) -> None:
    baseline, baseline_handoff = _handle(tmp_path, "baseline")
    candidate, candidate_handoff = _handle(tmp_path, "candidate")
    generation = StagedGenerationAdapter(
        service=EvaluationGenerationService(
            tmp_path / "artifacts",
            DeterministicFakeGenerationBackend(fallback=_output),
        ),
        baseline_run=baseline,
        candidate_run=candidate,
        baseline_handoff_ref=baseline_handoff,
        candidate_handoff_ref=candidate_handoff,
    )
    materialized = generation.execute(_request("generation_prompt_materialization"))
    baseline_step = generation.execute(_request("baseline_generation"))
    candidate_step = generation.execute(_request("candidate_generation"))
    assert materialized.status == "completed"
    assert baseline_step.records[0].kind == "generation_report"
    assert candidate_step.records[0].kind == "generation_report"
    assert (
        baseline_step.metadata["generation_settings_id"]
        == candidate_step.metadata["generation_settings_id"]
    )

    evaluation = StagedExperimentEvaluationAdapter(
        proposal=_proposal(),
        generation=generation,
        evaluator=ExperimentEvaluationService(),
    )
    checkpoint = evaluation.execute(_request("checkpoint_resolution"))
    comparison = evaluation.execute(_request("semantic_comparison"))
    deterministic = evaluation.execute(_request("deterministic_regression_evidence"))
    repeatability = evaluation.execute(_request("repeatability_variance_evidence"))
    review = evaluation.execute(_request("human_review_references"))
    assert checkpoint.metadata["checkpoint_ref"]
    assert comparison.records[0].kind == "experiment_comparison"
    assert deterministic.records[0].kind == "deterministic_regression_evidence"
    assert repeatability.records[0].kind == "repeatability_variance_evidence"
    assert review.records[0].kind == "human_review_references"


def test_transformers_generation_is_optional_and_network_free() -> None:
    capability = TransformersCausalLMGenerationBackend.capability()
    assert capability["remote_code"] is False
    assert capability["network_acquisition"] is False
    assert isinstance(capability["missing_packages"], list)
