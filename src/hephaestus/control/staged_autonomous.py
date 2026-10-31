"""Opt-in governed staged orchestration for autonomous experiments.

The module coordinates injected subsystem adapters.  It deliberately contains
no diagnosis, discovery, training, generation, evaluation, recovery, or judge
implementation.  The existing eight-phase spine remains the sole top-level
workflow order, while stable substep identities make pause/resume idempotent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from hephaestus.control.spine import SPINE_ORDER, SpinePhase
from hephaestus.control.staged_state import (
    GOVERNED_AUTONOMOUS_MODE,
    PHASE_SUBSTEPS,
    REQUIRED_RECORD_KINDS,
    RESUMABLE_STATUSES,
    STEP_STATUSES,
    TERMINAL_TRAINING_STATUSES,
    WORKFLOW_RECORD_COLLECTION,
    WORKFLOW_STATE_COLLECTION,
    StagedApprovalDecision,
    StagedApprovalRequest,
    StagedAutonomousDependencies,
    StagedAutonomousServices,
    StagedOperationRequest,
    StagedOperationResult,
    StagedOperationService,
    StagedOutputRecord,
    StagedStepState,
    StagedWorkflowState,
)
from hephaestus.policy.action_registry import evaluate_action_boundary
from hephaestus.schemas.external_operation import ExternalOperationPending
from hephaestus.schemas.operation_invocation import OperationInvocation
from hephaestus.state.execution_ownership import execution_owner


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(*parts: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:24]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.add(token)
            output.append(token)
    return output


class GovernedStagedOrchestrator:
    """Persistent, resumable coordinator for the opt-in autonomous path."""

    def __init__(
        self,
        *,
        workflow_id: str,
        run_id: str,
        lineage_id: str,
        stage_name: str,
        dependencies: StagedAutonomousDependencies,
    ) -> None:
        self.workflow_id = workflow_id
        self.run_id = run_id
        self.lineage_id = lineage_id
        self.stage_name = stage_name
        self.dependencies = dependencies

    def run(self, run_id: str | None = None) -> StagedWorkflowState:
        with execution_owner(self.dependencies.state_repository, "workflow", self.workflow_id):
            return self._run_owned(run_id)

    def _run_owned(self, run_id: str | None = None) -> StagedWorkflowState:
        if run_id is not None and run_id != self.run_id:
            raise ValueError("staged orchestrator run_id does not match its configured workflow")
        state = self._load_or_create()
        if state.completion_marker or (not state.resumable and state.status not in {"pending", "running"}):
            return state

        for phase in SPINE_ORDER:
            for substep in PHASE_SUBSTEPS[phase]:
                key = self._step_key(phase, substep)
                existing = state.steps.get(key)
                if existing is not None and existing.completion_marker:
                    continue
                state.current_phase = phase.value
                state.current_substep = substep
                state.status = "running"
                state.resumable = True  # A hard controller stop leaves an in-flight, reconcilable step.
                state.blocking_issues = []
                self._persist_state(state)

                step = self._execute_step(state, phase, substep, existing)
                state.steps[key] = step
                state.status = step.status
                state.blocking_issues = list(step.blocking_issues)
                state.resumable = step.resumable
                self._persist_state(state)
                if not step.completion_marker:
                    return state

            self._persist_record(
                phase=phase,
                substep="phase_completion",
                operation_id=f"phase-{_stable_id(self.workflow_id, phase.value)}",
                kind="phase_completion",
                payload={"phase": phase.value, "status": "completed"},
            )

        state.current_phase = SpinePhase.JUDGE_EXIT.value
        state.current_substep = "replay_evidence"
        state.status = "completed"
        state.resumable = False
        state.completion_marker = True
        self._persist_state(state)
        return state

    def resume(self) -> StagedWorkflowState:
        return self.run(self.run_id)

    def _load_or_create(self) -> StagedWorkflowState:
        payload = self.dependencies.state_repository.get_latest(
            WORKFLOW_STATE_COLLECTION, "workflow_id", self.workflow_id
        )
        if payload is not None:
            state = StagedWorkflowState.from_dict(payload)
            identity = (state.run_id, state.lineage_id, state.stage_name)
            expected = (self.run_id, self.lineage_id, self.stage_name)
            if identity != expected:
                raise ValueError("persisted staged workflow identity does not match configuration")
            if state.phase_order != [phase.value for phase in SPINE_ORDER]:
                raise ValueError("persisted staged workflow does not preserve SPINE_ORDER")
            return state
        state = StagedWorkflowState(
            workflow_id=self.workflow_id,
            run_id=self.run_id,
            lineage_id=self.lineage_id,
            stage_name=self.stage_name,
        )
        self._persist_state(state)
        return state

    def _execute_step(
        self,
        state: StagedWorkflowState,
        phase: SpinePhase,
        substep: str,
        existing: StagedStepState | None,
    ) -> StagedStepState:
        operation_id = (
            existing.operation_id
            if existing is not None
            else f"op-{_stable_id(self.workflow_id, phase.value, substep)}"
        )
        attempt = (existing.attempt if existing is not None else 0) + 1
        input_refs = self._input_refs(state)

        if substep in {"acquisition_approval_gate", "model_approval_gate", "action_approval_gate"}:
            result = self._approval_gate(state, phase, substep, operation_id)
        elif substep == "readiness_to_plan":
            evidence_failure = self._diagnosis_readiness_failure(state)
            result = evidence_failure or self._call_service(
                state, phase, substep, operation_id, attempt, input_refs
            )
        elif substep == "action_boundary":
            result = self._action_boundary(state)
        elif substep == "semantic_comparison":
            parity_failure = self._generation_parity_failure(state)
            result = parity_failure or self._call_service(state, phase, substep, operation_id, attempt, input_refs)
        elif substep == "action_application":
            authorization_failure = self._action_authorization_failure(state)
            result = authorization_failure or self._call_service(
                state, phase, substep, operation_id, attempt, input_refs
            )
        elif substep == "replay_evidence":
            result = self._replay_evidence(state)
        else:
            result = self._call_service(state, phase, substep, operation_id, attempt, input_refs)

        result = self._enforce_step_invariants(state, phase, substep, result)
        status = result.status if result.status in STEP_STATUSES else "terminal_failure"
        issues = list(result.blocking_issues)
        if result.status not in STEP_STATUSES:
            issues.append(f"invalid_subsystem_status:{result.status}")
        resumable = result.resumable if result.resumable is not None else status in RESUMABLE_STATUSES
        completed = status == "completed"

        for record in result.records:
            self._persist_record(
                phase=phase,
                substep=substep,
                operation_id=operation_id,
                kind=record.kind,
                payload=record.payload,
            )

        approval_request = existing.approval_request if existing is not None else None
        request_payload = result.metadata.get("approval_request")
        if isinstance(request_payload, dict):
            approval_request = dict(request_payload)
        output = dict(result.metadata)
        output["record_kinds"] = [record.kind for record in result.records]
        step = StagedStepState(
            phase=phase.value,
            substep=substep,
            operation_id=operation_id,
            status=status,
            attempt=attempt,
            input_refs=input_refs,
            output_refs=_unique(list(result.output_refs)),
            blocking_issues=_unique(issues),
            approval_request=approval_request,
            resumable=resumable,
            completion_marker=completed,
            output=output,
        )
        self._persist_record(
            phase=phase,
            substep=substep,
            operation_id=operation_id,
            kind="substep_result",
            payload={
                "status": step.status,
                "attempt": step.attempt,
                "resumable": step.resumable,
                "completion_marker": step.completion_marker,
                "input_refs": step.input_refs,
                "output_refs": step.output_refs,
                "blocking_issues": step.blocking_issues,
            },
        )
        return step

    def _call_service(self, state, phase, substep, operation_id, attempt, input_refs):
        """Persist intent before side effects; never replay an unknown outcome."""
        if self._service_for(phase, substep) is None:
            return self._invoke_service(state, phase, substep, operation_id, attempt, input_refs)
        repository = self.dependencies.state_repository
        prior = repository.get_latest("staged_operation_invocations", "operation_id", operation_id)
        if prior is not None and prior.get("status") == "started":
            return StagedOperationResult(status="blocked", resumable=True,
                blocking_issues=("ambiguous_operation_outcome",),
                metadata={"operation_id": operation_id,
                          "requires_independent_reconciliation": True})
        if prior is not None and int(prior["attempt"]) == attempt:
            payload = prior.get("result")
            if prior.get("result_ref"):
                artifacts = self.dependencies.artifact_store
                if artifacts is None or not artifacts.verify(prior["result_ref"]):
                    raise ValueError("staged invocation receipt is missing or corrupt")
                with artifacts.open(prior["result_ref"]) as stream:
                    payload = json.load(stream)
            if not isinstance(payload, dict):
                raise ValueError("staged invocation result is malformed")
            return StagedOperationResult(**{**payload,
                "output_refs": tuple(payload.get("output_refs", [])),
                "blocking_issues": tuple(payload.get("blocking_issues", [])),
                "records": tuple(StagedOutputRecord(**r) for r in payload.get("records", []))})
        intent = OperationInvocation(self.workflow_id, operation_id, attempt, "started")
        repository.append("staged_operation_invocations", intent.to_dict())
        result = self._invoke_service(state, phase, substep, operation_id, attempt, input_refs)
        intent.status = "returned"
        encoded = asdict(result)
        if self.dependencies.artifact_store is not None:
            intent.result_ref = self.dependencies.artifact_store.put_bytes(
                _canonical_json(encoded).encode(), media_type="application/json").artifact_ref
        else:
            intent.result = encoded
        repository.append("staged_operation_invocations", intent.to_dict())
        return result

    def _invoke_service(
        self,
        state: StagedWorkflowState,
        phase: SpinePhase,
        substep: str,
        operation_id: str,
        attempt: int,
        input_refs: list[str],
    ) -> StagedOperationResult:
        service = self._service_for(phase, substep)
        if service is None:
            return StagedOperationResult(
                status="blocked",
                blocking_issues=(f"service_not_configured:{phase.value}.{substep}",),
                resumable=True,
            )
        request = StagedOperationRequest(
            workflow_id=self.workflow_id,
            run_id=self.run_id,
            lineage_id=self.lineage_id,
            stage_name=self.stage_name,
            phase=phase.value,
            substep=substep,
            operation_id=operation_id,
            attempt=attempt,
            input_refs=tuple(input_refs),
            prior_outputs=self._prior_outputs(state),
        )
        try:
            return service.execute(request)
        except ExternalOperationPending as exc:
            return StagedOperationResult(
                status="interrupted" if exc.resumable else "blocked",
                resumable=exc.resumable,
                blocking_issues=("external_operation_outcome_requires_reconciliation",),
                records=(StagedOutputRecord("external_operation_pending", {
                    "operation_id":operation_id, "phase":phase.value, "substep":substep,
                    "error_type":type(exc).__name__, "resumable":exc.resumable}),),
            )
        except Exception as exc:  # subsystem exceptions are evidence, never success
            # Full details belong in private content-addressed custody, not
            # console logs or control memory. Preserve the original exception
            # before converting it into the terminal staged result.
            import traceback
            diagnostic = {
                "schema": "staged-subsystem-failure.v1",
                "workflow_id": self.workflow_id, "run_id": self.run_id,
                "lineage_id": self.lineage_id, "stage_name": self.stage_name,
                "operation_id": operation_id, "phase": phase.value,
                "substep": substep, "error_type": type(exc).__name__,
                "message": str(exc), "traceback": traceback.format_exc(),
            }
            diagnostic_ref = None
            diagnostic_capture_error = None
            if self.dependencies.artifact_store is not None:
                try:
                    diagnostic_ref = self.dependencies.artifact_store.put_bytes(
                        _canonical_json(diagnostic).encode("utf-8"),
                        media_type="application/json").artifact_ref
                except Exception as capture_error:
                    # Storage failure must not replace the original subsystem
                    # failure or accidentally turn it into a resumable result.
                    diagnostic_capture_error = type(capture_error).__name__
            return StagedOperationResult(
                status="terminal_failure",
                blocking_issues=(f"subsystem_exception:{type(exc).__name__}",),
                records=(
                    StagedOutputRecord(
                        "subsystem_failure",
                        {
                            "phase": phase.value,
                            "substep": substep,
                            "error_type": type(exc).__name__,
                            "diagnostic_ref": diagnostic_ref,
                            "diagnostic_capture_error": diagnostic_capture_error,
                            "retryable": False,
                        },
                    ),
                ),
                resumable=False,
            )

    def _service_for(self, phase: SpinePhase, substep: str) -> StagedOperationService | None:
        services = self.dependencies.services
        if phase is SpinePhase.JUDGE_ENTRY:
            return {
                "judge_entry_decision": services.judge_entry,
                "evidence_collection": services.evidence_collector,
                "evidence_based_diagnosis": services.diagnosis,
                "readiness_to_plan": services.plan_readiness,
            }[substep]
        if phase is SpinePhase.PLANNER:
            return services.planner
        if phase is SpinePhase.DATA_ACQUISITION_AUDIT:
            if substep == "dataset_discovery":
                return services.dataset_discovery
            if substep == "dataset_selection":
                return services.dataset_selection
            return services.dataset_acquisition
        if phase is SpinePhase.DATA_PREPROCESSOR:
            return services.data_preprocessor
        if phase is SpinePhase.TRAINING_ENGINEER:
            if substep == "model_discovery":
                return services.model_discovery
            if substep == "model_selection":
                return services.model_selection
            return services.training_lifecycle
        if phase is SpinePhase.RUNTIME_MONITOR:
            if substep == "training_status_poll":
                return services.training_lifecycle
            if substep == "bounded_recovery_advice":
                return services.recovery
            return services.runtime_monitor
        if phase is SpinePhase.EVALUATOR:
            if substep in {
                "generation_prompt_materialization",
                "baseline_generation",
                "candidate_generation",
            }:
                return services.generation
            return services.evaluator
        if phase is SpinePhase.JUDGE_EXIT:
            if substep == "action_application":
                return services.action_executor
            return services.judge_exit
        return None

    def _approval_gate(
        self,
        state: StagedWorkflowState,
        phase: SpinePhase,
        substep: str,
        operation_id: str,
    ) -> StagedOperationResult:
        source_key = {
            "acquisition_approval_gate": self._step_key(
                SpinePhase.DATA_ACQUISITION_AUDIT, "dataset_selection"
            ),
            "model_approval_gate": self._step_key(SpinePhase.TRAINING_ENGINEER, "model_selection"),
            "action_approval_gate": self._step_key(SpinePhase.JUDGE_EXIT, "action_boundary"),
        }[substep]
        source = state.steps.get(source_key)
        requirements = tuple(
            sorted(
                {
                    str(item)
                    for item in (source.output.get("required_approvals", []) if source else [])
                    if str(item).strip()
                }
            )
        )
        if not requirements:
            return StagedOperationResult(
                records=(
                    StagedOutputRecord(
                        "approval_gate",
                        {"required": False, "phase": phase.value, "substep": substep},
                    ),
                ),
                metadata={"approval_status": "not_required", "required_approvals": []},
            )
        subject_ref = (
            source.output.get("subject_ref") if source else None
        ) or (
            source.output_refs[0]
            if source and source.output_refs
            else source.operation_id
            if source
            else operation_id
        )
        request = StagedApprovalRequest(
            request_id=f"approval-{_stable_id(operation_id, subject_ref, requirements)}",
            workflow_id=self.workflow_id,
            run_id=self.run_id,
            operation_id=operation_id,
            phase=phase.value,
            substep=substep,
            subject_ref=str(subject_ref),
            requirements=requirements,
        )
        request_record = StagedOutputRecord("approval_request", request.to_dict())
        # Scientific eligibility is owned by the data service and cannot be
        # overridden by a permissive operator approval provider. No network IO
        # is permitted in this readiness check.
        if substep == "acquisition_approval_gate" and source and source.output.get("remote_dataset_review_required"):
            service = self.dependencies.services.dataset_acquisition
            readiness = getattr(service, "approval_readiness", None)
            issues = ("remote_dataset_review_service_missing",)
            if callable(readiness):
                try:
                    issues = readiness(StagedOperationRequest(
                        workflow_id=self.workflow_id, run_id=self.run_id,
                        lineage_id=self.lineage_id, stage_name=self.stage_name,
                        phase=phase.value, substep=substep, operation_id=operation_id,
                        attempt=1, input_refs=tuple(self._input_refs(state)),
                        prior_outputs={k: v.output for k, v in state.steps.items()}))
                    if not isinstance(issues, tuple) or any(not isinstance(v, str) or not v for v in issues):
                        issues = ("remote_dataset_readiness_invalid_result",)
                except Exception as exc:  # delegated scientific boundary: failure is never approval
                    issues = (f"remote_dataset_readiness_failed:{type(exc).__name__}",)
            if issues:
                return StagedOperationResult(status="approval_pending", records=(request_record,),
                    blocking_issues=tuple(issues), resumable=True,
                    metadata={"approval_status": "pending",
                        "approval_request": request.to_dict(),
                        "required_approvals": list(requirements)})
        decision = (
            self.dependencies.approval_service.decision_for(request)
            if self.dependencies.approval_service is not None
            else None
        )
        if decision is None:
            return StagedOperationResult(
                status="approval_pending",
                records=(request_record,),
                blocking_issues=("approval_pending",),
                metadata={
                    "approval_request": request.to_dict(),
                    "required_approvals": list(requirements),
                    "approval_status": "pending",
                },
                resumable=True,
            )
        mismatch = []
        if decision.request_id != request.request_id:
            mismatch.append("request_id")
        if decision.operation_id != request.operation_id:
            mismatch.append("operation_id")
        if decision.subject_ref != request.subject_ref:
            mismatch.append("subject_ref")
        if decision.requirements and tuple(sorted(decision.requirements)) != requirements:
            mismatch.append("requirements")
        if mismatch:
            return StagedOperationResult(
                status="blocked",
                records=(request_record, StagedOutputRecord("approval_decision", decision.to_dict())),
                blocking_issues=(f"stale_or_mismatched_approval:{','.join(mismatch)}",),
                metadata={
                    "approval_request": request.to_dict(),
                    "approval_status": "mismatched",
                    "required_approvals": list(requirements),
                },
                resumable=False,
            )
        if decision.status != "approved" or not decision.approval_ref:
            return StagedOperationResult(
                status="blocked",
                records=(request_record, StagedOutputRecord("approval_decision", decision.to_dict())),
                blocking_issues=(f"approval_{decision.status}",),
                metadata={
                    "approval_request": request.to_dict(),
                    "approval_status": decision.status,
                    "required_approvals": list(requirements),
                },
                resumable=False,
            )
        return StagedOperationResult(
            records=(request_record, StagedOutputRecord("approval_decision", decision.to_dict())),
            output_refs=(decision.approval_ref,),
            metadata={
                "approval_request": request.to_dict(),
                "approval_status": "approved",
                "approval_ref": decision.approval_ref,
                "required_approvals": list(requirements),
            },
        )

    def _action_boundary(self, state: StagedWorkflowState) -> StagedOperationResult:
        verdict = state.steps.get(self._step_key(SpinePhase.JUDGE_EXIT, "governed_verdict"))
        action = str((verdict.output if verdict else {}).get("proposed_action", "")).strip()
        boundary = evaluate_action_boundary(action)
        requirements: list[str] = []
        if boundary["requires_approval"]:
            requirements = [
                "operator_high_risk_approval"
                if boundary["high_risk"]
                else "operator_approval"
            ]
        metadata = {
            "proposed_action": action,
            "subject_ref": str((verdict.output if verdict else {}).get("subject_ref", action)),
            "required_approvals": requirements,
            "action_boundary": boundary,
        }
        record = StagedOutputRecord("action_boundary", {"action": action, **boundary})
        if not boundary["known_action"] or boundary["forbidden"]:
            return StagedOperationResult(
                status="blocked",
                records=(record,),
                blocking_issues=(
                    "unknown_action" if not boundary["known_action"] else "forbidden_action",
                ),
                metadata=metadata,
                resumable=False,
            )
        return StagedOperationResult(records=(record,), metadata=metadata)

    def _generation_parity_failure(self, state: StagedWorkflowState) -> StagedOperationResult | None:
        baseline = state.steps.get(self._step_key(SpinePhase.EVALUATOR, "baseline_generation"))
        candidate = state.steps.get(self._step_key(SpinePhase.EVALUATOR, "candidate_generation"))
        baseline_settings = str((baseline.output if baseline else {}).get("generation_settings_id", ""))
        candidate_settings = str((candidate.output if candidate else {}).get("generation_settings_id", ""))
        if not baseline_settings or not candidate_settings or baseline_settings != candidate_settings:
            return StagedOperationResult(
                status="blocked",
                records=(
                    StagedOutputRecord(
                        "generation_parity_failure",
                        {
                            "baseline_generation_settings_id": baseline_settings,
                            "candidate_generation_settings_id": candidate_settings,
                        },
                    ),
                ),
                blocking_issues=("generation_parity_mismatch",),
                resumable=False,
            )
        return None

    def _diagnosis_readiness_failure(
        self, state: StagedWorkflowState
    ) -> StagedOperationResult | None:
        diagnosis = state.steps.get(
            self._step_key(SpinePhase.JUDGE_ENTRY, "evidence_based_diagnosis")
        )
        missing = [
            str(item)
            for item in (diagnosis.output.get("missing_evidence", []) if diagnosis else [])
            if str(item).strip()
        ]
        blocking = bool((diagnosis.output if diagnosis else {}).get("blocking_diagnosis", False))
        if not missing and not blocking:
            return None
        return StagedOperationResult(
            status="inconclusive",
            records=(
                StagedOutputRecord(
                    "planning_readiness",
                    {
                        "ready": False,
                        "missing_evidence": missing,
                        "blocking_diagnosis": blocking,
                    },
                ),
            ),
            blocking_issues=("diagnosis_evidence_insufficient",),
            metadata={"missing_evidence": missing, "ready_to_plan": False},
            resumable=True,
        )

    def _action_authorization_failure(
        self, state: StagedWorkflowState
    ) -> StagedOperationResult | None:
        boundary_step = state.steps.get(self._step_key(SpinePhase.JUDGE_EXIT, "action_boundary"))
        approval_step = state.steps.get(self._step_key(SpinePhase.JUDGE_EXIT, "action_approval_gate"))
        promotion_step = state.steps.get(self._step_key(SpinePhase.JUDGE_EXIT, "promotion_gate"))
        action = str((boundary_step.output if boundary_step else {}).get("proposed_action", ""))
        approval_status = str(
            (approval_step.output if approval_step else {}).get("approval_status", "not_required")
        )
        boundary = evaluate_action_boundary(action, context={"approval_status": approval_status})
        if not boundary["allowed"]:
            return StagedOperationResult(
                status="blocked",
                blocking_issues=("action_not_authorized",),
                records=(
                    StagedOutputRecord(
                        "action_application_blocked",
                        {"action": action, "action_boundary": boundary},
                    ),
                ),
                resumable=False,
            )
        gate_output = promotion_step.output if promotion_step else {}
        protected_actions = {
            "promote_checkpoint",
            "rollback_to_checkpoint",
            "branch_new_experiment",
            "restart_lineage",
        }
        gate_allowed = bool(
            gate_output.get(
                "action_allowed",
                gate_output.get("promotion_allowed", False),
            )
        )
        if action in protected_actions and not gate_allowed:
            return StagedOperationResult(
                status="blocked",
                blocking_issues=("governed_action_gate_blocked",),
                records=(
                    StagedOutputRecord(
                        "action_application_blocked",
                        {"action": action, "reason": "governed_action_gate_blocked"},
                    ),
                ),
                resumable=False,
            )
        return None

    def _replay_evidence(self, state: StagedWorkflowState) -> StagedOperationResult:
        outputs = self._prior_outputs(state)
        replay = {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
            "stage_name": self.stage_name,
            "phase_order": [phase.value for phase in SPINE_ORDER],
            "operation_ids": [step.operation_id for step in state.steps.values()],
            "dataset_revision": self._first_output(outputs, "dataset_revision"),
            "model_revision": self._first_output(outputs, "model_revision"),
            "config_identity": self._first_output(outputs, "config_identity"),
            "seed_identity": self._first_output(outputs, "seed_identity"),
            "checkpoint_ref": self._first_output(outputs, "checkpoint_ref"),
            "generation_settings_id": self._first_output(outputs, "generation_settings_id"),
            "eval_pack_id": self._first_output(outputs, "eval_pack_id"),
            "approval_refs": _unique(
                [
                    str(output.get("approval_ref", ""))
                    for output in outputs.values()
                    if output.get("approval_ref")
                ]
            ),
            "evidence_refs": self._input_refs(state),
        }
        required = (
            "dataset_revision",
            "model_revision",
            "config_identity",
            "seed_identity",
            "checkpoint_ref",
            "generation_settings_id",
            "eval_pack_id",
        )
        missing = [key for key in required if not replay.get(key)]
        record = StagedOutputRecord("replay_metadata", replay)
        if missing:
            return StagedOperationResult(
                status="inconclusive",
                records=(record,),
                blocking_issues=(f"replay_evidence_incomplete:{','.join(missing)}",),
                metadata={"replay_complete": False, "missing_replay_evidence": missing},
                resumable=True,
            )
        return StagedOperationResult(
            records=(record,),
            output_refs=tuple(replay["evidence_refs"]),
            metadata={"replay_complete": True, **replay},
        )

    def _enforce_step_invariants(
        self,
        state: StagedWorkflowState,
        phase: SpinePhase,
        substep: str,
        result: StagedOperationResult,
    ) -> StagedOperationResult:
        expected_kind = REQUIRED_RECORD_KINDS.get(substep)
        if (
            result.status == "completed"
            and expected_kind is not None
            and all(record.kind != expected_kind for record in result.records)
        ):
            return StagedOperationResult(
                status="terminal_failure",
                records=result.records,
                output_refs=result.output_refs,
                blocking_issues=(
                    *result.blocking_issues,
                    f"decision_critical_record_missing:{expected_kind}",
                ),
                metadata=result.metadata,
                resumable=False,
            )
        if substep == "lifecycle_launch" and result.status == "completed":
            training_status = str(result.metadata.get("training_status", ""))
            if not training_status:
                return StagedOperationResult(
                    status="terminal_failure",
                    records=result.records,
                    output_refs=result.output_refs,
                    blocking_issues=(*result.blocking_issues, "launch_missing_training_status"),
                    metadata=result.metadata,
                    resumable=False,
                )
        if substep == "training_status_poll" and result.status == "completed":
            training_status = str(result.metadata.get("training_status", ""))
            if training_status not in TERMINAL_TRAINING_STATUSES:
                return StagedOperationResult(
                    status="interrupted",
                    records=result.records,
                    output_refs=result.output_refs,
                    blocking_issues=(*result.blocking_issues, f"training_not_terminal:{training_status or 'unknown'}"),
                    metadata=result.metadata,
                    resumable=True,
                )
            if training_status == "failed":
                return StagedOperationResult(
                    status="terminal_failure",
                    records=result.records,
                    output_refs=result.output_refs,
                    blocking_issues=(*result.blocking_issues, "training_failed"),
                    metadata=result.metadata,
                    resumable=False,
                )
            if training_status == "cancelled":
                return StagedOperationResult(
                    status="cancelled",
                    records=result.records,
                    output_refs=result.output_refs,
                    blocking_issues=result.blocking_issues,
                    metadata=result.metadata,
                    resumable=False,
                )
        if substep == "checkpoint_resolution" and result.status == "completed":
            checkpoint_ref = str(result.metadata.get("checkpoint_ref", "")).strip()
            if not checkpoint_ref:
                return StagedOperationResult(
                    status="blocked",
                    records=result.records,
                    output_refs=result.output_refs,
                    blocking_issues=(*result.blocking_issues, "concrete_checkpoint_required"),
                    metadata=result.metadata,
                    resumable=False,
                )
            if (
                checkpoint_ref.startswith("sha256:")
                and self.dependencies.artifact_store is not None
                and not self.dependencies.artifact_store.verify(checkpoint_ref)
            ):
                return StagedOperationResult(
                    status="blocked",
                    records=result.records,
                    output_refs=result.output_refs,
                    blocking_issues=(*result.blocking_issues, "checkpoint_integrity_failed"),
                    metadata=result.metadata,
                    resumable=False,
                )
        if substep == "action_application" and result.status == "completed":
            boundary_step = state.steps.get(self._step_key(SpinePhase.JUDGE_EXIT, "action_boundary"))
            gate_step = state.steps.get(self._step_key(SpinePhase.JUDGE_EXIT, "action_approval_gate"))
            action = str((boundary_step.output if boundary_step else {}).get("proposed_action", ""))
            approval_status = str((gate_step.output if gate_step else {}).get("approval_status", "not_required"))
            boundary = evaluate_action_boundary(action, context={"approval_status": approval_status})
            if not boundary["allowed"]:
                return StagedOperationResult(
                    status="blocked",
                    records=result.records,
                    output_refs=result.output_refs,
                    blocking_issues=(*result.blocking_issues, "action_not_authorized"),
                    metadata=result.metadata,
                    resumable=False,
                )
        return result

    def _persist_state(self, state: StagedWorkflowState) -> None:
        self.dependencies.state_repository.append(WORKFLOW_STATE_COLLECTION, state.to_dict())

    def _persist_record(
        self,
        *,
        phase: SpinePhase,
        substep: str,
        operation_id: str,
        kind: str,
        payload: dict[str, object],
    ) -> None:
        record_id = f"record-{_stable_id(operation_id, kind, _canonical_json(payload))}"
        existing = self.dependencies.state_repository.get_latest(
            WORKFLOW_RECORD_COLLECTION, "record_id", record_id
        )
        if existing is not None:
            return
        rows = self.dependencies.state_repository.all(WORKFLOW_RECORD_COLLECTION)
        sequence = 1 + max(
            [
                int(row.get("sequence", 0))
                for row in rows
                if row.get("workflow_id") == self.workflow_id
            ]
            or [0]
        )
        wrapped = {
            "record_id": record_id,
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
            "stage_name": self.stage_name,
            "phase": phase.value,
            "substep": substep,
            "operation_id": operation_id,
            "sequence": sequence,
            "kind": kind,
            "payload": payload,
        }
        if self.dependencies.artifact_store is not None:
            from hephaestus.storage.verified import canonical_bytes
            artifact = self.dependencies.artifact_store.put_bytes(canonical_bytes(wrapped), media_type="application/json")
            wrapped["artifact_ref"] = artifact.artifact_ref
        self.dependencies.state_repository.append(WORKFLOW_RECORD_COLLECTION, wrapped)
        if self.dependencies.record_sink is not None:
            self.dependencies.record_sink.append(kind, payload)

    def _step_key(self, phase: SpinePhase, substep: str) -> str:
        return f"{phase.value}.{substep}"

    def _input_refs(self, state: StagedWorkflowState) -> list[str]:
        refs: list[str] = []
        for phase in SPINE_ORDER:
            for substep in PHASE_SUBSTEPS[phase]:
                step = state.steps.get(self._step_key(phase, substep))
                if step is not None:
                    refs.extend(step.output_refs)
        return _unique(refs)

    def _prior_outputs(self, state: StagedWorkflowState) -> dict[str, dict[str, object]]:
        return {
            key: dict(step.output)
            for key, step in state.steps.items()
            if step.completion_marker or step.status in RESUMABLE_STATUSES
        }

    @staticmethod
    def _first_output(outputs: dict[str, dict[str, object]], key: str) -> object | None:
        for output in outputs.values():
            value = output.get(key)
            if value not in (None, "", [], {}):
                return value
        return None


def build_staged_autonomous_orchestrator(
    *,
    run_id: str,
    lineage_id: str,
    stage_name: str,
    dependencies: StagedAutonomousDependencies,
    workflow_id: str | None = None,
) -> GovernedStagedOrchestrator:
    return GovernedStagedOrchestrator(
        workflow_id=workflow_id or f"workflow-{run_id}",
        run_id=run_id,
        lineage_id=lineage_id,
        stage_name=stage_name,
        dependencies=dependencies,
    )
