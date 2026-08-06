"""Concrete generation/evaluation adapters for staged orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from hephaestus.control.staged_state import (
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
    comparison: ExperimentComparison | None = field(default=None, init=False)

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
            if self.generation.baseline_result is None or self.generation.candidate_result is None:
                return StagedOperationResult(
                    status="blocked",
                    blocking_issues=("generation_evidence_missing",),
                    resumable=True,
                )
            baseline = self.generation.baseline_result.run_handle
            candidate = self.generation.candidate_result.run_handle
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
            return StagedOperationResult(
                status="blocked",
                blocking_issues=("experiment_comparison_missing",),
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
