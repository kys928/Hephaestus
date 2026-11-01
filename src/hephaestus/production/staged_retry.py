"""Bridge terminal zero-update training failures to a fresh bounded attempt.

No failed workflow is edited or called completed. Ambiguous/post-update work,
evaluation failures, OOM, cancellations and approvals cannot use this bridge.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

from hephaestus.control.staged_state import StagedOperationResult, WORKFLOW_RECORD_COLLECTION
from hephaestus.policy.staged_retry import scientific_fingerprint
from hephaestus.recovery.models import RecoveryRequest, RecoveryAttempt
from hephaestus.schemas.experiment_contract import ExperimentProposal
from hephaestus.schemas.staged_retry import StagedRetry
from hephaestus.schemas.optimizer_progress import OptimizerProgress
from hephaestus.state.staged_retry_store import StagedRetryStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from .staged_records import verified_workflow_payload


def _sealed_record(runtime, workflow_id, substep, kind):
    rows = [r for r in runtime.state_repository.all(WORKFLOW_RECORD_COLLECTION)
            if r.get("workflow_id") == workflow_id and r.get("substep") == substep and r.get("kind") == kind]
    if not rows:
        raise ValueError("retry requires original sealed training records")
    row = rows[-1]
    if read_verified_json(runtime.artifact_store, row["artifact_ref"]) != {k: v for k, v in row.items() if k != "artifact_ref"}:
        raise ValueError("retry source record differs from its immutable receipt")
    return row


@dataclass(slots=True)
class RetryInvariantTrainingService:
    delegate: object
    runtime: object
    expected_fingerprint: str
    cycle_key: str

    def _validate(self, req):
        if req.substep == "lifecycle_launch":
            source = verified_workflow_payload(self.runtime.state_repository, self.runtime.artifact_store, req,
                substep="training_input_binding", kind="experiment_training_binding")
            if scientific_fingerprint(ExperimentProposal.from_dict(source)) != self.expected_fingerprint:
                return StagedOperationResult(status="blocked", resumable=False,
                    blocking_issues=("operational_retry_changed_scientific_variables",))
            current = StagedRetryStore(self.runtime.state_repository).current(self.cycle_key)
            if current is not None and current.decision_ref:
                decision = read_verified_json(self.runtime.artifact_store, current.decision_ref)
                attempt = self.runtime.recovery.attempt_store.get(decision["attempt_id"])
                if attempt is not None and attempt.status == "planned":
                    attempt.status = "executing"
                    self.runtime.recovery.attempt_store.update(attempt)
        return None

    def execute(self, req):
        blocked = self._validate(req)
        if blocked is not None:
            return blocked
        return self.delegate.execute(req)

    def reconcile_operation(self, req):
        blocked = self._validate(req)
        if blocked is not None:
            return blocked
        method = getattr(self.delegate, "reconcile_operation", None)
        if not callable(method):
            return StagedOperationResult(status="blocked", resumable=True,
                blocking_issues=("retry_requires_original_provider_reconciliation",))
        return method(req)


def finish_staged_retry(runtime, current, status):
    if current.decision_ref:
        decision = read_verified_json(runtime.artifact_store, current.decision_ref)
        attempt = runtime.recovery.attempt_store.get(decision["attempt_id"])
        if attempt is not None and attempt.status in {"planned", "executing"}:
            attempt.status = status
            runtime.recovery.attempt_store.update(attempt)


def prepare_staged_retry(runtime, state, progress, current: StagedRetry):
    """Return a sealed eligible decision, or None. No launch occurs here."""
    if (current.attempt >= current.maximum_attempts or progress.resumable
            or progress.current_substep not in {"lifecycle_launch", "training_status_poll"}
            or progress.status != "terminal_failure"):
        return None
    try:
        source = _sealed_record(runtime, progress.workflow_id, progress.current_substep, "training_run_handle")
        if any(source.get(k) != expected for k, expected in {"run_id": progress.run_id,
                "lineage_id": state.lineage_id, "stage_name": state.stage_name}.items()):
            return None
        handle = runtime.training_lifecycle.status(progress.run_id)
        if (source["payload"] != handle.to_dict() or handle.status != "failed"
                or handle.checkpoint_refs or handle.resume_token_ref
                or not handle.issues or any(i.code != "non_zero_exit" for i in handle.issues)):
            return None
        ref = Path(handle.metadata["optimizer_progress_ref"])
        raw = ref.read_bytes()
        if "sha256:" + sha256(raw).hexdigest() != handle.metadata["optimizer_progress_hash"]:
            return None
        evidence = OptimizerProgress.from_dict(json.loads(raw)).to_dict()
        if (evidence.get("schema") != "hf-optimizer-progress.v1" or evidence.get("run_id") != progress.run_id
                or evidence.get("config_fingerprint") != handle.metadata["config_fingerprint"]
                or type(evidence.get("optimizer_steps")) is not int or evidence["optimizer_steps"] != 0
                or evidence.get("optimizer_update_started") is not False or evidence.get("phase") != "pre_optimizer"):
            return None
        proposal = ExperimentProposal.from_dict(_sealed_record(runtime, progress.workflow_id,
            "training_input_binding", "experiment_training_binding")["payload"])
        if (proposal.run_id, proposal.lineage_id, proposal.stage_name) != (progress.run_id, state.lineage_id, state.stage_name):
            return None
        progress_ref = runtime.artifact_store.put_bytes(raw).artifact_ref
        failure_ref = runtime.artifact_store.put_bytes(canonical_bytes({
            "schema": "staged-zero-update-failure.v1", "handle_ref": source["artifact_ref"],
            "optimizer_progress_ref": progress_ref, "failed_workflow_id": progress.workflow_id})).artifact_ref
        decision = runtime.recovery.decide(RecoveryRequest(
            "bootstrap-recovery-"+progress.run_id, progress.run_id, handle.experiment_id,
            state.lineage_id, state.stage_name, current.cycle_key,
            evidence=[{"evidence_kind": "training_run_handle", "source_ref": failure_ref,
                "code": "non_zero_exit", "confidence": .95, "status": "failed"}],
            constraints={"progress_observed": False, "budget_window_id": current.cycle_key},
            requested_action_kind="restart_bounded_job"))
        decision_ref = runtime.artifact_store.put_bytes(canonical_bytes(decision.to_dict())).artifact_ref
        if (decision.status != "eligible" or decision.recommendation.registry_action != "rerun_same_config"
                or decision.recommendation.approval_required):
            return None
        next_attempt = StagedRetry(current.cycle_key, current.attempt+1,
            f"{state.program_id}-cycle-{state.cycle_index+1}-attempt-{current.attempt+1}",
            current.maximum_attempts, current.run_id, failure_ref, decision_ref, scientific_fingerprint(proposal),
            (datetime.now(timezone.utc)+timedelta(seconds=decision.backoff.delay_seconds)).isoformat())
        runtime.recovery.attempt_store.record(RecoveryAttempt(
            decision.attempt_id, decision.request_id, decision.run_id, decision.experiment_id,
            decision.lineage_id, current.cycle_key, decision.classification.failure_signature,
            decision.classification.evidence_fingerprint, decision.recommendation.action_kind,
            decision.recommendation.registry_action, "planned", input_evidence_refs=[failure_ref],
            result_ref=decision_ref, metadata={"budget_window_id": current.cycle_key, "scientific_variables_changed": False}))
        StagedRetryStore(runtime.state_repository).save(next_attempt)
        return next_attempt
    except (KeyError, ValueError, OSError, PermissionError, TypeError):
        return None
