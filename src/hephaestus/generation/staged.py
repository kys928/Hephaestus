"""Concrete generation/evaluation adapters for staged orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from hephaestus.control.staged_state import (
    WORKFLOW_RECORD_COLLECTION,
    StagedOperationRequest,
    StagedOperationResult,
    StagedOutputRecord,
)
from hephaestus.evaluation.experiment_service import ExperimentEvaluationService
from hephaestus.schemas.experiment_contract import (
    ExperimentComparison,
    ExperimentProposal,
    TrainingRunHandle,
)
from hephaestus.storage.base import StateRepository

from .models import GenerationResult
from .service import EvaluationGenerationService


@dataclass(slots=True)
class StagedGenerationAdapter:
    """Materialize frozen prompts and generate baseline/candidate evidence."""

    service: EvaluationGenerationService
    baseline_run: TrainingRunHandle
    candidate_run: TrainingRunHandle
    baseline_handoff_ref: str | None = None
    candidate_handoff_ref: str | None = None
    baseline_result: GenerationResult | None = field(default=None, init=False)
    candidate_result: GenerationResult | None = field(default=None, init=False)

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        if request.substep == "generation_prompt_materialization":
            plan = self.service.plan()
            return StagedOperationResult(
                records=(
                    StagedOutputRecord(
                        "generation_prompt_manifest",
                        {
                            "eval_pack_id": plan.eval_pack_id,
                            "eval_pack_version": plan.eval_pack_version,
                            "content_hash": plan.content_hash,
                            "generation_settings_id": plan.generation_settings_id,
                            "seed_identity": plan.seed_identity,
                            "task_seed_count": len(plan.tasks),
                        },
                    ),
                ),
                metadata={
                    "eval_pack_id": plan.eval_pack_id,
                    "eval_pack_version": plan.eval_pack_version,
                    "generation_settings_id": plan.generation_settings_id,
                    "seed_identity": plan.seed_identity,
                    "content_hash": plan.content_hash,
                },
            )
        if request.substep == "baseline_generation":
            self.baseline_result = self.service.generate(
                self.baseline_run,
                generation_handoff_ref=self.baseline_handoff_ref,
            )
            return self._operation_result(self.baseline_result)
        if request.substep == "candidate_generation":
            self.candidate_result = self.service.generate(
                self.candidate_run,
                generation_handoff_ref=self.candidate_handoff_ref,
            )
            return self._operation_result(self.candidate_result)
        return StagedOperationResult(
            status="blocked",
            blocking_issues=(f"unsupported_generation_substep:{request.substep}",),
            resumable=False,
        )

    @staticmethod
    def _operation_result(result: GenerationResult) -> StagedOperationResult:
        report = result.report
        status = "completed" if report.completed else (
            "retryable_failure"
            if any(issue.retryable for issue in report.issues)
            else "blocked"
        )
        return StagedOperationResult(
            status=status,
            records=(StagedOutputRecord("generation_report", report.to_dict()),),
            output_refs=tuple(
                ref for ref in [report.report_ref, *report.evidence_refs] if ref
            ),
            blocking_issues=tuple(issue.code for issue in report.issues if issue.blocking),
            metadata={
                "run_id": report.run_id,
                "checkpoint_ref": report.checkpoint_ref,
                "generation_settings_id": report.generation_settings_id,
                "seed_identity": report.seed_identity,
                "eval_pack_id": report.eval_pack_id,
                "eval_pack_version": report.eval_pack_version,
                "generation_complete": report.completed,
                "training_run_handle": result.run_handle.to_dict(),
            },
            resumable=status == "retryable_failure",
        )


@dataclass(slots=True)
class StagedExperimentEvaluationAdapter:
    """Resolve generated handles and expose comparison evidence by evaluator substep."""

    proposal: ExperimentProposal
    generation: StagedGenerationAdapter
    evaluator: ExperimentEvaluationService
    record_repository: StateRepository | None = None
    comparison: ExperimentComparison | None = field(default=None, init=False)

    def _completed_handle(
        self, request: StagedOperationRequest, substep: str, expected_run_id: str
    ) -> TrainingRunHandle | None:
        previous = request.prior_outputs.get(f"evaluator.{substep}", {})
        if previous.get("generation_complete") is not True:
            return None
        value = previous.get("training_run_handle")
        if not isinstance(value, dict):
            return None
        try:
            run = TrainingRunHandle.from_dict(value)
        except (KeyError, TypeError, ValueError):
            return None
        if run.run_id != expected_run_id or run.status != "completed":
            return None
        if not run.checkpoint_refs or run.checkpoint_refs[-1] != previous.get("checkpoint_ref"):
            return None
        if (run.metadata.get("semantic_evaluation") is None or
            run.metadata.get("generation_report") is None):
            return None
        return run

    def _restore_comparison(self, request: StagedOperationRequest) -> ExperimentComparison | None:
        prior = request.prior_outputs.get("evaluator.semantic_comparison", {})
        identity = str(prior.get("comparison_id") or "").strip()
        if not identity or self.record_repository is None:
            return None
        for row in reversed(self.record_repository.all(WORKFLOW_RECORD_COLLECTION)):
            if (row.get("workflow_id") != request.workflow_id or
                row.get("substep") != "semantic_comparison" or
                row.get("kind") != "experiment_comparison"):
                continue
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("comparison_id") != identity:
                continue
            comparison = ExperimentComparison.from_dict(payload)
            if comparison.experiment_id != self.proposal.experiment_id:
                return None
            return comparison
        return None

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        if request.substep == "checkpoint_resolution":
            checkpoint = (
                self.generation.candidate_run.checkpoint_refs[-1]
                if self.generation.candidate_run.checkpoint_refs
                else ""
            )
            if not checkpoint:
                return StagedOperationResult(
                    status="blocked",
                    blocking_issues=("candidate_checkpoint_missing",),
                    resumable=False,
                )
            return StagedOperationResult(
                records=(
                    StagedOutputRecord(
                        "checkpoint_resolution",
                        {"run_id": self.generation.candidate_run.run_id, "checkpoint_ref": checkpoint},
                    ),
                ),
                output_refs=(checkpoint,),
                metadata={"checkpoint_ref": checkpoint},
            )
        if request.substep == "semantic_comparison":
            baseline = (
                self.generation.baseline_result.run_handle
                if self.generation.baseline_result is not None
                else self._completed_handle(request, "baseline_generation", self.generation.baseline_run.run_id)
            )
            candidate = (
                self.generation.candidate_result.run_handle
                if self.generation.candidate_result is not None
                else self._completed_handle(request, "candidate_generation", self.generation.candidate_run.run_id)
            )
            if baseline is None or candidate is None:
                return StagedOperationResult(
                    status="blocked",
                    blocking_issues=("verified_generation_handle_missing",),
                    resumable=True,
                )
            proposal = ExperimentProposal(**self.proposal.to_dict())
            proposal.baseline_ref = baseline.run_id
            self.comparison = self.evaluator.compare(proposal, [baseline, candidate])
            status = (
                "blocked"
                if self.comparison.primary_outcome == "invalid_comparison"
                or any(issue.blocking for issue in self.comparison.issues)
                else "completed"
            )
            return StagedOperationResult(
                status=status,
                records=(
                    StagedOutputRecord("experiment_comparison", self.comparison.to_dict()),
                ),
                output_refs=tuple(self.comparison.evidence_refs),
                blocking_issues=tuple(
                    issue.code for issue in self.comparison.issues if issue.blocking
                ),
                metadata={
                    "comparison_id": self.comparison.comparison_id,
                    "primary_outcome": self.comparison.primary_outcome,
                    "recommendation": self.comparison.recommendation,
                    "eval_pack_id": self.comparison.metadata.get("eval_pack_id"),
                },
                resumable=status != "completed",
            )
        if self.comparison is None:
            self.comparison = self._restore_comparison(request)
        if self.comparison is None:
            return StagedOperationResult(
                status="blocked",
                blocking_issues=("verified_experiment_comparison_missing",),
                resumable=True,
            )
        if request.substep == "deterministic_regression_evidence":
            return StagedOperationResult(
                records=(
                    StagedOutputRecord(
                        "deterministic_regression_evidence",
                        {
                            "comparison_id": self.comparison.comparison_id,
                            "deterministic_gate_status": self.comparison.deterministic_gate_status,
                            "hard_failures": self.comparison.effect_summary.get("deterministic", {}).get(
                                "candidate_hard_failures", []
                            ),
                        },
                    ),
                ),
                metadata={"deterministic_gate_status": self.comparison.deterministic_gate_status},
            )
        if request.substep == "repeatability_variance_evidence":
            return StagedOperationResult(
                records=(
                    StagedOutputRecord(
                        "repeatability_variance_evidence",
                        {
                            "comparison_id": self.comparison.comparison_id,
                            "variance_risk": self.comparison.variance_risk,
                            "repeatability": self.comparison.effect_summary.get("repeatability", {}),
                        },
                    ),
                ),
                metadata={"variance_risk": self.comparison.variance_risk},
            )
        if request.substep == "human_review_references":
            review = self.comparison.metadata.get("human_review_bundle", {})
            return StagedOperationResult(
                records=(
                    StagedOutputRecord(
                        "human_review_references",
                        {
                            "comparison_id": self.comparison.comparison_id,
                            "human_review_bundle": review,
                        },
                    ),
                ),
                metadata={"human_review_bundle": review},
            )
        return StagedOperationResult(
            status="blocked",
            blocking_issues=(f"unsupported_evaluation_substep:{request.substep}",),
            resumable=False,
        )
