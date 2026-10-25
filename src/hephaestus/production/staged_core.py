"""Evidence-bound adapters for the production 45-operation scientific spine.

These are narrow adapters over existing diagnosis, training, recovery and Judge
services. A request cannot supply its own trusted facts via arbitrary metadata.
A completed stage is an evidence claim, not a certification or promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.control.staged_state import (
    StagedAutonomousServices, StagedOperationRequest, StagedOperationResult,
    StagedOutputRecord,
)
from hephaestus.recovery.models import RecoveryRequest
from hephaestus.schemas.diagnosis_contract import DiagnosisRequest, DiagnosisReport
from hephaestus.schemas.experiment_contract import ExperimentComparison
from hephaestus.storage.verified import read_verified_json
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest

from .staged_records import verified_workflow_payload

_ENTRY = frozenset({
    "judge_entry_decision", "evidence_collection", "evidence_based_diagnosis",
    "readiness_to_plan",
})
_MONITOR = frozenset({
    "runtime_observation", "runtime_evidence", "runtime_control_governance",
})
_JUDGE = frozenset({"governed_verdict", "promotion_gate"})


def _bad(code: str, *, resumable: bool = False) -> StagedOperationResult:
    return StagedOperationResult(
        status="inconclusive" if resumable else "blocked",
        blocking_issues=(code,), resumable=resumable,
    )


@dataclass(slots=True)
class StagedScientificEntry:
    """Read only operator-pinned source facts and run the actual diagnosis."""

    artifacts: object
    diagnosis_service: object
    diagnosis_request_ref: str

    def _source(self, req: StagedOperationRequest):
        payload = read_verified_json(self.artifacts, self.diagnosis_request_ref)
        if payload.get("schema") != "staged-diagnosis-input.v1":
            raise ValueError("diagnosis input has no versioned contract")
        for key in ("run_id", "lineage_id", "stage_name"):
            if payload.get(key) != getattr(req, key):
                raise PermissionError("diagnosis input was pinned for a different scientific run")
        evidence_refs = payload.get("evidence_refs")
        if (not isinstance(evidence_refs, list) or not evidence_refs
                or len(set(evidence_refs)) != len(evidence_refs)):
            raise ValueError("diagnosis requires independently pinned evidence references")
        rows = []
        for ref in evidence_refs:
            if not isinstance(ref, str) or not ref.startswith("sha256:"):
                raise ValueError("unverified diagnosis evidence reference")
            item = read_verified_json(self.artifacts, ref)
            if (not isinstance(item, dict)
                    or item.get("run_id") not in (None, req.run_id)
                    or item.get("lineage_id") not in (None, req.lineage_id)
                    or not item.get("evidence_kind")):
                raise ValueError("diagnostic source evidence lacks kind or has wrong scope")
            rows.append({**item, "source_ref": ref})
        return payload, rows

    def execute(self, req: StagedOperationRequest) -> StagedOperationResult:
        if req.substep not in _ENTRY:
            return _bad("unsupported_scientific_entry_operation")
        try:
            source, rows = self._source(req)
            if req.substep == "judge_entry_decision":
                return StagedOperationResult(
                    records=(StagedOutputRecord("judge_entry", {
                        "source_ref": self.diagnosis_request_ref,
                        "run_id": req.run_id, "decision": "proceed_to_verified_diagnosis",
                        "certification_state": "not_certified",
                    }),), metadata={"entry_evidence_ref": self.diagnosis_request_ref})
            if req.substep == "evidence_collection":
                return StagedOperationResult(
                    records=(StagedOutputRecord("evidence_bundle", {
                        "diagnosis_request_ref": self.diagnosis_request_ref,
                        "evidence_refs": source["evidence_refs"],
                        "evidence_count": len(rows),
                    }),), output_refs=tuple(source["evidence_refs"]),
                    metadata={"evidence_count": len(rows)})
            if req.substep == "evidence_based_diagnosis":
                bundled = req.prior_outputs.get("judge_entry.evidence_collection", {})
                if bundled.get("evidence_count") != len(rows):
                    raise ValueError("evidence collection not verified before diagnosis")
                diagnosis_request = DiagnosisRequest(
                    request_id=f"diagnosis-{req.operation_id}",
                    run_id=req.run_id, lineage_id=req.lineage_id,
                    stage_name=req.stage_name, observed_failures=rows,
                    evidence_refs=list(source["evidence_refs"]),
                    constraints={"input_ref": self.diagnosis_request_ref},
                )
                result = self.diagnosis_service.diagnose(diagnosis_request)
                if (result.run_id != req.run_id or result.lineage_id != req.lineage_id
                        or not result.report_id):
                    raise ValueError("diagnosis returned unrelated scientific source")
                return StagedOperationResult(
                    records=(StagedOutputRecord("diagnosis_report", result.to_dict()),),
                    metadata={"diagnosis_report_id": result.report_id,
                              "diagnosis_status": result.status,
                              "leading_hypothesis_id": result.leading_hypothesis_id,
                              "missing_evidence": list(result.missing_evidence),
                              "blocking_diagnosis": any(i.blocking for i in result.issues)})
            report = DiagnosisReport.from_dict(verified_workflow_payload(
                self.diagnosis_service_repository(req), self.artifacts, req,
                substep="evidence_based_diagnosis", kind="diagnosis_report"))
            if report.missing_evidence or any(i.blocking for i in report.issues):
                return _bad("diagnosis_not_ready_for_planning", resumable=True)
            return StagedOperationResult(
                records=(StagedOutputRecord("planning_readiness", {
                    "diagnosis_report_id": report.report_id, "ready": True,
                    "source_ref": self.diagnosis_request_ref,
                }),), metadata={"ready_to_plan": True})
        except (ValueError, TypeError, PermissionError, KeyError, OSError) as exc:
            return _bad(f"scientific_entry_invalid:{type(exc).__name__}")

    # Repository assigned explicitly so the service cannot trust caller metadata.
    repository: object = None

    def diagnosis_service_repository(self, req):
        if self.repository is None:
            raise ValueError("scientific entry requires production state repository")
        return self.repository


@dataclass(slots=True)
class StagedPlannerHandoff:
    repository: object
    artifacts: object

    def execute(self, req: StagedOperationRequest) -> StagedOperationResult:
        if req.substep != "diagnosis_handoff":
            return _bad("unsupported_planner_handoff")
        try:
            report = DiagnosisReport.from_dict(verified_workflow_payload(
                self.repository, self.artifacts, req,
                substep="evidence_based_diagnosis", kind="diagnosis_report"))
            if (report.run_id != req.run_id or report.lineage_id != req.lineage_id
                    or report.stage_name != req.stage_name
                    or report.missing_evidence or any(x.blocking for x in report.issues)):
                return _bad("planner_handoff_has_unresolved_diagnosis")
            return StagedOperationResult(records=(StagedOutputRecord(
                "diagnosis_handoff", {
                    "diagnosis_report_id": report.report_id,
                    "leading_hypothesis_id": report.leading_hypothesis_id,
                    "confidence": report.confidence,
                    "run_id": req.run_id,
                }),), metadata={"diagnosis_report_id": report.report_id})
        except (ValueError, PermissionError, TypeError, KeyError, OSError) as exc:
            return _bad(f"planner_handoff_invalid:{type(exc).__name__}")


@dataclass(slots=True)
class StagedRuntimeObservation:
    """Observe persisted trainer state, never infer health from launch alone."""

    lifecycle: object

    def _handle(self, req):
        receipt = req.prior_outputs.get("runtime_monitor.training_status_poll", {})
        if receipt.get("training_status") != "completed" or receipt.get("run_id") != req.run_id:
            raise ValueError("no verified completed training poll")
        handle = self.lifecycle.status(req.run_id)
        if (handle.status != "completed" or handle.run_id != req.run_id
                or not handle.checkpoint_refs
                or handle.checkpoint_refs[-1] != receipt.get("checkpoint_ref")
                or handle.experiment_id != receipt.get("experiment_id")):
            raise ValueError("persisted training lifecycle differs from poll evidence")
        checkpoint = handle.checkpoint_refs[-1]
        valid, digest = validate_checkpoint_manifest(Path(checkpoint))
        if not valid:
            raise ValueError("actual training checkpoint failed verification")
        return handle, checkpoint, digest

    def execute(self, req: StagedOperationRequest) -> StagedOperationResult:
        if req.substep not in _MONITOR:
            return _bad("unsupported_runtime_monitor_stage")
        try:
            handle, checkpoint, digest = self._handle(req)
            if req.substep == "runtime_observation":
                payload = {
                    "run_id": req.run_id, "experiment_id": handle.experiment_id,
                    "training_status": handle.status, "checkpoint_ref": checkpoint,
                    "checkpoint_manifest_hash": digest,
                    "metrics_ref": handle.metrics_ref,
                }
                return StagedOperationResult(
                    records=(StagedOutputRecord("runtime_observation", payload),),
                    metadata={"runtime_health": "verified_completed",
                              "checkpoint_ref": checkpoint})
            if req.prior_outputs.get("runtime_monitor.runtime_observation", {}).get(
                    "runtime_health") != "verified_completed":
                raise ValueError("unobserved runtime cannot publish evidence")
            if req.substep == "runtime_evidence":
                return StagedOperationResult(
                    records=(StagedOutputRecord("runtime_evidence", {
                        "run_id": req.run_id, "experiment_id": handle.experiment_id,
                        "checkpoint_ref": checkpoint, "checkpoint_manifest_hash": digest,
                        "event_stream_ref": handle.event_stream_ref,
                        "metrics_ref": handle.metrics_ref,
                        "observed_status": handle.status,
                    }),), metadata={"runtime_evidence_complete": True,
                                  "monitor_outcome": "healthy"})
            if not req.prior_outputs.get(
                    "runtime_monitor.runtime_evidence", {}).get("runtime_evidence_complete"):
                raise ValueError("no runtime evidence for governance")
            return StagedOperationResult(
                records=(StagedOutputRecord("runtime_control_recommendation", {
                    "run_id": req.run_id,
                    "recommendation": "no_control_intervention",
                    "training_status": handle.status,
                    "operation_executed": False,
                }),), metadata={"runtime_control_action_executed": False})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return _bad(f"runtime_observation_invalid:{type(exc).__name__}")


@dataclass(slots=True)
class StagedRecoveryAdvice:
    """Invoke policy engine only on actual incident evidence; otherwise none."""

    service: object
    repository: object
    artifacts: object

    def execute(self, req: StagedOperationRequest) -> StagedOperationResult:
        if req.substep != "bounded_recovery_advice":
            return _bad("unsupported_recovery_substep")
        try:
            observed = verified_workflow_payload(
                self.repository, self.artifacts, req,
                substep="runtime_evidence", kind="runtime_evidence")
            if observed.get("run_id") != req.run_id:
                raise ValueError("runtime evidence belongs to another run")
            incidents = observed.get("incidents", [])
            if not isinstance(incidents, list):
                raise ValueError("invalid runtime incident inventory")
            if not incidents and observed.get("observed_status") == "completed":
                return StagedOperationResult(
                    records=(StagedOutputRecord("recovery_decision", {
                        "run_id": req.run_id, "status": "not_required",
                        "basis": "verified_completed_training_no_reported_incidents",
                        "recovery_executed": False,
                    }),), metadata={"recovery_required": False})
            decision = self.service.decide(RecoveryRequest(
                request_id=f"recovery-{req.operation_id}",
                run_id=req.run_id, lineage_id=req.lineage_id,
                experiment_id=str(observed["experiment_id"]),
                stage_name=req.stage_name, operation_id=req.operation_id,
                evidence=incidents, constraints={"execution_allowed": False},
            ))
            return StagedOperationResult(
                records=(StagedOutputRecord("recovery_decision", decision.to_dict()),),
                metadata={"recovery_status": decision.status,
                          "recovery_executed": False})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return _bad(f"recovery_advice_invalid:{type(exc).__name__}")


@dataclass(slots=True)
class StagedSemanticJudge:
    """Run existing Judge policy over immutable comparison, with no promotion."""

    repository: object
    artifacts: object
    judge: object

    def execute(self, req: StagedOperationRequest) -> StagedOperationResult:
        if req.substep not in _JUDGE:
            return _bad("unsupported_judge_substep")
        try:
            comparison = ExperimentComparison.from_dict(verified_workflow_payload(
                self.repository, self.artifacts, req,
                substep="semantic_comparison", kind="experiment_comparison"))
            checkpoint = req.prior_outputs.get("evaluator.checkpoint_resolution", {}).get(
                "checkpoint_ref")
            if not checkpoint or not comparison.comparison_id:
                raise ValueError("Judge has no verified comparison/checkpoint")
            if req.substep == "governed_verdict":
                outcome = self.judge.decide(
                    comparison, run_id=req.run_id, lineage_id=req.lineage_id,
                    candidate_checkpoint_ref=str(checkpoint),
                )
                action = outcome.next_action.value
                # Even a promising tiny-model effect cannot become an implicit
                # certificate or promotion: conservative model of this gate.
                if action == "promote_checkpoint":
                    action = "request_recheck"
                return StagedOperationResult(
                    records=(StagedOutputRecord("judge_exit", outcome.to_dict()),),
                    metadata={"proposed_action": action, "subject_ref": checkpoint,
                              "confidence": outcome.confidence,
                              "certification_state": "not_certified"})
            action = req.prior_outputs.get("judge_exit.action_boundary", {}).get(
                "proposed_action")
            # Promotion is always denied at this stage without an independently
            # issued scientific certificate, and approval is NEVER generated.
            return StagedOperationResult(
                records=(StagedOutputRecord("promotion_gate_report", {
                    "action": action, "promotion_allowed": False,
                    "certification_state": "not_certified",
                    "reason": "no_independent_scientific_certificate",
                }),), metadata={"promotion_allowed": False,
                              "action_allowed": action not in {
                                  "promote_checkpoint", "branch_new_experiment",
                                  "restart_lineage", "rollback_to_checkpoint",
                              },
                              "certification_state": "not_certified"})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return _bad(f"semantic_judge_invalid:{type(exc).__name__}")


def make_staged_core_services(runtime, *, diagnosis_request_ref: str) -> StagedAutonomousServices:
    """Production adapters only; all unrelated stages remain explicitly unset."""
    entry = StagedScientificEntry(
        artifacts=runtime.artifact_store,
        diagnosis_service=runtime.diagnosis_service,
        diagnosis_request_ref=diagnosis_request_ref,
        repository=runtime.state_repository,
    )
    return StagedAutonomousServices(
        judge_entry=entry, evidence_collector=entry,
        diagnosis=entry, plan_readiness=entry,
        planner=StagedPlannerHandoff(runtime.state_repository, runtime.artifact_store),
        runtime_monitor=StagedRuntimeObservation(runtime.training_lifecycle),
        recovery=StagedRecoveryAdvice(
            runtime.recovery, runtime.state_repository, runtime.artifact_store),
        judge_exit=StagedSemanticJudge(
            runtime.state_repository, runtime.artifact_store, runtime.judge),
        evaluator=StagedMissingEvaluation(),
    )


class StagedMissingEvaluation:
    """Explicit fail-closed dependency, replaced by real evaluation composition."""

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        return _bad("evaluation_not_yet_composed")
