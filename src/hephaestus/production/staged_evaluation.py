"""Exact baseline + verified completed candidate -> real generation and scoring.

Owns no science, no model inference implementation and no action decisions.
Rehydrates immutable campaign inputs on every operation, and delegates to
the existing frozen-pack generation and deterministic evaluation services.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.control.staged_state import (
    StagedOperationRequest, StagedOperationResult,
)
from hephaestus.evaluation.experiment_service import ExperimentEvaluationService
from hephaestus.generation.staged import (
    StagedExperimentEvaluationAdapter, StagedGenerationAdapter,
)
from hephaestus.generation.service import EvaluationGenerationService
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingRunHandle
from hephaestus.storage.base import StateRepository, ArtifactStore
from hephaestus.storage.verified import read_verified_json
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest

from .staged_records import verified_workflow_payload


_GENERATION = frozenset({
    "generation_prompt_materialization", "baseline_generation", "candidate_generation",
})
_EVALUATION = frozenset({
    "semantic_comparison", "deterministic_regression_evidence",
    "repeatability_variance_evidence", "human_review_references",
})


@dataclass(slots=True)
class StagedVerifiedEvaluation:
    repository: StateRepository
    artifacts: ArtifactStore
    lifecycle: object
    generation_service: EvaluationGenerationService
    evaluation_service: ExperimentEvaluationService
    checkpoint_service: object
    baseline_run_artifact_ref: str
    inference_enabled: bool = False

    def _inputs(self, request: StagedOperationRequest):
        data = verified_workflow_payload(
            self.repository, self.artifacts, request,
            substep="training_input_binding", kind="experiment_training_binding",
        )
        proposal = ExperimentProposal(**data)
        if (proposal.run_id != request.run_id or proposal.lineage_id != request.lineage_id
                or proposal.stage_name != request.stage_name or not proposal.experiment_id):
            raise ValueError("final experiment proposal does not match scientific workflow")
        ref = self.baseline_run_artifact_ref
        baseline_source = read_verified_json(self.artifacts, ref)
        if baseline_source.get("schema") != "verified-baseline-run.v1":
            raise ValueError("baseline reference must resolve to a pinned verified run handle")
        baseline = TrainingRunHandle.from_dict(baseline_source["handle"])
        if (baseline.status != "completed" or not baseline.run_id
                or not baseline.checkpoint_refs or
                proposal.baseline_ref not in (baseline.run_id, baseline.checkpoint_refs[-1])):
            raise ValueError("experiment baseline does not match verified pinned parent")
        # The loading-instructions paths are consumed/verified again by the
        # existing generation service when it actually loads model weights.
        if not baseline.metadata.get("generation_handoff_ref") and not baseline.metadata.get("loading_instructions_ref"):
            raise ValueError("frozen baseline is missing loading instructions")
        candidate = self.lifecycle.status(request.run_id)
        completed = request.prior_outputs.get("runtime_monitor.training_status_poll", {})
        resolved = request.prior_outputs.get("evaluator.checkpoint_resolution", {})
        if (candidate.status != "completed"
                or candidate.run_id != request.run_id
                or candidate.experiment_id != proposal.experiment_id
                or not candidate.checkpoint_refs
                or completed.get("training_status") != "completed"
                or completed.get("checkpoint_ref") != candidate.checkpoint_refs[-1]
                or not candidate.metadata.get("generation_handoff_ref")
                    and not candidate.metadata.get("loading_instructions_ref")):
            raise ValueError("candidate lacks completed matching training and generation handoff")
        checkpoint = candidate.checkpoint_refs[-1]
        verified, manifest_sha = validate_checkpoint_manifest(Path(checkpoint))
        if not verified:
            raise ValueError(f"candidate checkpoint failed content verification: {manifest_sha}")
        # checkpoint_resolution happens before prompt/materialization and is
        # cross-checked if already present in the workflow.
        if resolved and (resolved.get("checkpoint_ref") != checkpoint
                         or resolved.get("checkpoint_manifest_hash") != manifest_sha):
            raise ValueError("evaluator checkpoint and training lifecycle disagree")
        generation = StagedGenerationAdapter(
            self.generation_service, baseline, candidate,
        )
        evaluation = StagedExperimentEvaluationAdapter(
            proposal, generation, self.evaluation_service,
            record_repository=self.repository,
        )
        return generation, evaluation

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        if request.substep == "checkpoint_resolution":
            return self.checkpoint_service.execute(request)
        if request.substep not in _GENERATION | _EVALUATION:
            return StagedOperationResult(
                status="blocked", resumable=False,
                blocking_issues=(f"unsupported_generation_or_evaluation_stage:{request.substep}",),
            )
        if not self.inference_enabled:
            return StagedOperationResult(
                status="approval_pending", resumable=True,
                blocking_issues=("real_generation_disabled",),
            )
        try:
            generation, evaluation = self._inputs(request)
            return (generation if request.substep in _GENERATION else evaluation).execute(request)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return StagedOperationResult(
                status="blocked", resumable=False,
                blocking_issues=(f"verified_evaluation_handoff_invalid:{type(exc).__name__}",),
            )
