from __future__ import annotations

from copy import deepcopy

import pytest

from hephaestus.policy.recovery_policy import RecoveryPolicy
from hephaestus.recovery import (
    BoundedRecoveryService,
    FakeRecoveryActionExecutor,
    InMemoryRecoveryAttemptStore,
    RecoveryActionExecutor,
    RecoveryAttempt,
    RecoveryAttemptStore,
    RecoveryController,
    RecoveryDecision,
    RecoveryRequest,
)
from hephaestus.roles.incident_responder import IncidentResponderRole


def _request(
    *evidence: dict[str, object],
    request_id: str = "recovery-1",
    operation_id: str = "operation-1",
    stage_name: str = "smoke_test",
    requested_action_kind: str | None = None,
    approval_evidence: list[dict[str, object]] | None = None,
    constraints: dict[str, object] | None = None,
) -> RecoveryRequest:
    return RecoveryRequest(
        request_id=request_id,
        run_id="run-1",
        experiment_id="experiment-1",
        lineage_id="lineage-1",
        stage_name=stage_name,
        operation_id=operation_id,
        evidence=list(evidence),
        requested_action_kind=requested_action_kind,
        approval_evidence=approval_evidence or [],
        constraints=constraints or {},
    )


def _service(
    store: InMemoryRecoveryAttemptStore | None = None,
    policy: RecoveryPolicy | None = None,
) -> BoundedRecoveryService:
    return BoundedRecoveryService(
        attempt_store=store or InMemoryRecoveryAttemptStore(),
        policy=policy or RecoveryPolicy(),
    )


def _evidence(
    signal: str, ref: str = "evidence:1", confidence: float = 0.9
) -> dict[str, object]:
    return {
        "kind": "fixture",
        "source_ref": ref,
        "signals": [signal],
        "confidence": confidence,
    }


def _checkpoint_bundle(
    *,
    hash_verified: bool = True,
    token_valid: bool = True,
    replay_status: str = "reproducible",
    token_overrides: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    compatibility = {
        "model_revision": "model-v1",
        "tokenizer_ref": "tokenizer:v1",
        "architecture_family": "decoder-v1",
        "training_recipe_ref": "recipe:v1",
        "data_contract_ref": "data:v1",
        "data_contract_hash": "data-hash",
        "backend_id": "local-fixture",
    }
    token_compatibility = {**compatibility, **(token_overrides or {})}
    checkpoint: dict[str, object] = {
        "kind": "checkpoint_record",
        "source_ref": "checkpoint-record:1",
        "checkpoint_ref": "checkpoint:stable-1",
        "exists": True,
        "integrity_level": "content_hash_verified"
        if hash_verified
        else "reference_only",
        "content_hash": "checkpoint-hash" if hash_verified else "",
        "hash_verified": hash_verified,
        "resume_compatibility": compatibility,
    }
    token: dict[str, object] = {
        "kind": "resume_token",
        "source_ref": "resume-token:1",
        "resume_token_ref": "resume-token:1",
        "checkpoint_ref": "checkpoint:stable-1",
        "exists": True,
        "valid": token_valid,
        "compatibility": token_compatibility,
    }
    replay = {
        "kind": "replay_verification_report",
        "source_ref": "replay:1",
        "status": replay_status,
    }
    lineage = {
        "kind": "lineage_state",
        "source_ref": "lineage:1",
        "status": "promising",
        "trust_level": "medium",
    }
    return [checkpoint, token, replay, lineage]


def _approval(action: str) -> dict[str, object]:
    return {
        "decision_event_id": f"approval:{action}",
        "run_id": "run-1",
        "lineage_id": "lineage-1",
        "status": "approved",
        "effect_on_action": "execute_requested_action",
        "metadata": {"proposed_action": action},
    }


def test_transient_provider_failure_is_retryable_and_backoff_is_deterministic() -> None:
    service = _service()
    request = _request(_evidence("provider_unavailable"))

    first = service.decide(request)
    second = service.decide(deepcopy(request))

    assert first.classification.category == "transient_provider_outage"
    assert first.classification.retryability == "retryable"
    assert first.classification.safe_to_automate is True
    assert first.recommendation.action_kind == "retry_after_backoff"
    assert first.status == "eligible"
    assert first.backoff.to_dict() == second.backoff.to_dict()
    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize(
    ("signal", "category"),
    [
        ("provider_unavailable", "transient_provider_outage"),
        ("network_interruption", "transient_network_download_interruption"),
        ("lease_expired", "worker_lease_loss"),
        ("process_crash", "process_crash"),
        ("explicit_cancellation", "explicit_cancellation"),
        ("operator_interruption", "operator_interruption"),
        ("out_of_memory", "out_of_memory"),
        ("budget_exhausted", "resource_budget_exhaustion"),
        ("data_loader_failure", "data_loader_failure"),
        ("malformed_data", "malformed_or_contaminated_data"),
        ("tokenizer_incompatible", "tokenizer_incompatibility"),
        ("model_checkpoint_incompatible", "model_checkpoint_incompatibility"),
        ("checkpoint_corrupt", "checkpoint_corruption"),
        ("checkpoint_missing", "missing_checkpoint_evidence"),
        ("resume_token_corrupt", "resume_token_corruption"),
        ("metrics_missing", "missing_metrics"),
        ("evaluation_incomplete", "incomplete_evaluation"),
        ("deterministic_regression", "deterministic_regression"),
        ("high_evaluation_variance", "high_evaluation_variance"),
        ("replay_failed", "replay_failure"),
        ("policy_blocked", "policy_or_approval_block"),
        ("invalid_configuration", "invalid_configuration"),
        ("unsupported_capability", "permanent_unsupported_capability"),
        ("lineage_poisoned", "poisoned_or_deprecated_lineage"),
        ("storage_integrity_failure", "storage_integrity_failure"),
        ("state_persistence_failure", "state_persistence_failure"),
    ],
)
def test_finite_failure_taxonomy_is_classifiable(signal: str, category: str) -> None:
    decision = _service().decide(_request(_evidence(signal)))

    assert decision.classification.category == category


def test_permanent_incompatibility_is_not_retryable() -> None:
    decision = _service().decide(_request(_evidence("tokenizer_incompatible")))

    assert decision.classification.category == "tokenizer_incompatibility"
    assert decision.classification.retryability == "not_retryable"
    assert decision.recommendation.action_kind == "stop"


def test_nested_diagnosis_observation_signals_are_consumed_without_guessing() -> None:
    diagnosis = {
        "kind": "diagnosis_report",
        "source_ref": "diagnosis:1",
        "leading_hypothesis_id": "hyp-runtime",
        "hypotheses": [
            {
                "hypothesis_id": "hyp-runtime",
                "failure_domain": "runtime_or_system",
            }
        ],
        "observations": [
            {
                "source_ref": "incident:loader",
                "metadata": {"signals": ["data_loader_failure"]},
            }
        ],
        "confidence": 0.9,
    }

    decision = _service().decide(_request(diagnosis))

    assert decision.classification.category == "data_loader_failure"
    assert decision.classification.evidence_refs == ["diagnosis:1"]


def test_corrupted_checkpoint_refuses_requested_resume() -> None:
    bundle = _checkpoint_bundle(hash_verified=False)
    request = _request(
        _evidence("checkpoint_corrupt"),
        *bundle,
        requested_action_kind="resume_verified_checkpoint",
    )

    decision = _service().decide(request)

    assert decision.status == "blocked"
    assert decision.checkpoint is not None and decision.checkpoint.allowed is False
    assert decision.recommendation.action_kind != "resume_verified_checkpoint"
    assert {issue.code for issue in decision.checkpoint.issues} >= {
        "missing_checkpoint_content_hash",
        "checkpoint_hash_unverified",
    }


def test_missing_checkpoint_hash_refuses_resume() -> None:
    bundle = _checkpoint_bundle()
    bundle[0]["content_hash"] = ""
    bundle[0]["hash_verified"] = False
    bundle[0]["integrity_level"] = "reference_only"
    decision = _service().decide(
        _request(
            _evidence("process_crash"),
            *bundle,
            requested_action_kind="resume_verified_checkpoint",
        )
    )

    assert decision.status == "blocked"
    assert decision.checkpoint is not None and decision.checkpoint.allowed is False


def test_valid_checkpoint_allows_resume_recommendation() -> None:
    decision = _service().decide(
        _request(
            _evidence("process_crash"),
            *_checkpoint_bundle(),
            requested_action_kind="resume_verified_checkpoint",
        )
    )

    assert decision.checkpoint is not None and decision.checkpoint.allowed is True
    assert decision.recommendation.action_kind == "resume_verified_checkpoint"
    assert decision.recommendation.registry_action == "continue_from_checkpoint"
    assert decision.status == "eligible"


def test_checkpoint_compatibility_mismatch_refuses_resume() -> None:
    decision = _service().decide(
        _request(
            _evidence("process_crash"),
            *_checkpoint_bundle(token_overrides={"tokenizer_ref": "tokenizer:v2"}),
            requested_action_kind="resume_verified_checkpoint",
        )
    )

    assert decision.checkpoint is not None and decision.checkpoint.allowed is False
    assert any(
        issue.code == "resume_tokenizer_ref_mismatch"
        for issue in decision.checkpoint.issues
    )


def test_partial_replay_evidence_blocks_resume() -> None:
    decision = _service().decide(
        _request(
            _evidence("process_crash"),
            *_checkpoint_bundle(replay_status="partial"),
            requested_action_kind="resume_verified_checkpoint",
        )
    )

    assert decision.checkpoint is not None and decision.checkpoint.allowed is False
    assert any(
        issue.code == "replay_policy_blocks_resume"
        for issue in decision.checkpoint.issues
    )


def test_worker_lease_loss_produces_bounded_replacement_behavior() -> None:
    decision = _service().decide(_request(_evidence("lease_expired")))

    assert decision.classification.category == "worker_lease_loss"
    assert decision.recommendation.action_kind == "request_replacement_worker"
    assert decision.recommendation.parameters["require_exclusive_new_lease"] is True
    assert decision.budget.allowed is True


def test_stale_worker_completion_is_rejected_not_accepted_as_progress() -> None:
    decision = _service().decide(
        _request(_evidence("lease_expired"), _evidence("stale_result", "worker:late"))
    )

    assert decision.recommendation.action_kind == "collect_more_evidence"
    assert decision.recommendation.parameters["stale_completion_rejected"] is True
    assert decision.recommendation.parameters["exclusive_ownership_required"] is True


def test_duplicate_worker_ownership_blocks_replacement_until_resolved() -> None:
    decision = _service().decide(
        _request(
            _evidence("lease_expired"), _evidence("duplicate_ownership", "worker:dup")
        )
    )

    assert decision.recommendation.action_kind == "collect_more_evidence"
    assert decision.recommendation.action_kind != "request_replacement_worker"


def test_retry_budget_exhaustion_stops_identical_retries() -> None:
    store = InMemoryRecoveryAttemptStore()
    service = _service(store)
    request = _request(_evidence("provider_unavailable"))
    first = service.decide(request)
    store.record(
        RecoveryAttempt(
            attempt_id=first.attempt_id,
            request_id=first.request_id,
            run_id=first.run_id,
            experiment_id=first.experiment_id,
            lineage_id=first.lineage_id,
            operation_id="operation-1",
            failure_signature=first.classification.failure_signature,
            evidence_fingerprint=first.classification.evidence_fingerprint,
            action_kind=first.recommendation.action_kind,
            registry_action=first.recommendation.registry_action,
            status="failed",
        )
    )

    second = service.decide(request)

    assert second.budget.allowed is False
    assert "identical_evidence_without_progress" in second.budget.reasons
    assert second.recommendation.action_kind == "stop"


def test_new_evidence_permits_reconsideration_within_remaining_budget() -> None:
    store = InMemoryRecoveryAttemptStore()
    service = _service(store)
    first = service.decide(_request(_evidence("provider_unavailable")))
    store.record(
        RecoveryAttempt(
            attempt_id=first.attempt_id,
            request_id=first.request_id,
            run_id=first.run_id,
            experiment_id=first.experiment_id,
            lineage_id=first.lineage_id,
            operation_id="operation-1",
            failure_signature=first.classification.failure_signature,
            evidence_fingerprint=first.classification.evidence_fingerprint,
            action_kind=first.recommendation.action_kind,
            registry_action=first.recommendation.registry_action,
            status="failed",
        )
    )

    reconsidered = service.decide(
        _request(
            _evidence("provider_unavailable"),
            _evidence("provider_unavailable", ref="provider:new"),
            request_id="recovery-2",
        )
    )

    assert reconsidered.budget.allowed is True
    assert reconsidered.recommendation.action_kind == "retry_after_backoff"


def test_high_impact_rollback_requires_matching_approval() -> None:
    constraints = {
        "verified_rollback_checkpoint_ref": "checkpoint:stable",
        "verified_rollback_checkpoint_hash": "sha256:stable",
    }
    missing = _service().decide(
        _request(
            _evidence("deterministic_regression"),
            stage_name="stabilization",
            constraints=constraints,
        )
    )
    approved = _service().decide(
        _request(
            _evidence("deterministic_regression"),
            stage_name="stabilization",
            constraints=constraints,
            approval_evidence=[_approval("rollback_to_checkpoint")],
        )
    )

    assert missing.recommendation.action_kind == "rollback_verified_checkpoint"
    assert missing.status == "approval_required"
    assert approved.status == "eligible"
    assert approved.recommendation.approval_ref == "approval:rollback_to_checkpoint"


def test_missing_approval_prevents_controller_execution() -> None:
    constraints = {
        "verified_rollback_checkpoint_ref": "checkpoint:stable",
        "verified_rollback_checkpoint_hash": "sha256:stable",
    }
    decision = _service().decide(
        _request(
            _evidence("deterministic_regression"),
            stage_name="stabilization",
            constraints=constraints,
        )
    )
    fake = FakeRecoveryActionExecutor()
    result = RecoveryController(
        InMemoryRecoveryAttemptStore(),
        {"rollback_verified_checkpoint": fake},
        {"rollback_verified_checkpoint"},
    ).execute(decision)

    assert result.status == "blocked"
    assert fake.calls == []


def test_approved_high_impact_action_executes_only_through_injected_handler() -> None:
    constraints = {
        "verified_rollback_checkpoint_ref": "checkpoint:stable",
        "verified_rollback_checkpoint_hash": "sha256:stable",
    }
    store = InMemoryRecoveryAttemptStore()
    decision = _service(store).decide(
        _request(
            _evidence("deterministic_regression"),
            stage_name="stabilization",
            constraints=constraints,
            approval_evidence=[_approval("rollback_to_checkpoint")],
        )
    )
    fake = FakeRecoveryActionExecutor()
    controller = RecoveryController(
        store,
        {"rollback_verified_checkpoint": fake},
        {"rollback_verified_checkpoint"},
    )

    result = controller.execute(decision)

    assert decision.status == "eligible"
    assert result.status == "succeeded"
    assert fake.calls == [(decision.attempt_id, "rollback_verified_checkpoint")]


def test_low_impact_allowlisted_action_executes_through_injected_handler() -> None:
    store = InMemoryRecoveryAttemptStore()
    decision = _service(store).decide(_request(_evidence("provider_unavailable")))
    fake = FakeRecoveryActionExecutor()
    controller = RecoveryController(
        store,
        {"retry_after_backoff": fake},
        {"retry_after_backoff"},
    )

    result = controller.execute(decision)
    repeated = controller.execute(decision)

    assert result.status == "succeeded"
    assert repeated.status == "already_succeeded"
    assert len(fake.calls) == 1


def test_unknown_action_is_blocked() -> None:
    decision = _service().decide(
        _request(
            _evidence("provider_unavailable"),
            requested_action_kind="invent_unregistered_recovery",
        )
    )

    assert decision.status == "blocked"
    assert any(issue.code == "unknown_recovery_action" for issue in decision.issues)


def test_poisoned_lineage_requires_approval_to_quarantine() -> None:
    lineage = {
        "kind": "lineage_state",
        "source_ref": "lineage:poisoned",
        "status": "poisoned",
        "trust_level": "low",
    }
    missing = _service().decide(_request(lineage))
    approved = _service().decide(
        _request(lineage, approval_evidence=[_approval("quarantine_lineage")])
    )

    assert missing.recommendation.action_kind == "quarantine_lineage"
    assert missing.status == "approval_required"
    assert approved.status == "eligible"


def test_high_evaluation_variance_requests_more_evidence_not_rollback() -> None:
    comparison = {
        "kind": "experiment_comparison",
        "source_ref": "comparison:1",
        "primary_outcome": "inconclusive",
        "deterministic_gate_status": "incomplete",
        "variance_risk": "high",
        "confidence": 0.9,
    }
    decision = _service().decide(_request(comparison))

    assert decision.classification.category == "high_evaluation_variance"
    assert decision.recommendation.action_kind == "rerun_evaluation"
    assert decision.recommendation.action_kind != "rollback_verified_checkpoint"


def test_deterministic_regression_is_never_ignored() -> None:
    constraints = {
        "verified_rollback_checkpoint_ref": "checkpoint:stable",
        "verified_rollback_checkpoint_hash": "sha256:stable",
    }
    decision = _service().decide(
        _request(
            _evidence("deterministic_regression"),
            _evidence("high_evaluation_variance", "variance:1"),
            stage_name="stabilization",
            constraints=constraints,
        )
    )

    assert decision.classification.category == "deterministic_regression"
    assert decision.recommendation.action_kind == "rollback_verified_checkpoint"


def test_conflicting_evidence_returns_inconclusive_and_blocks_automation() -> None:
    decision = _service().decide(
        _request(
            _evidence("provider_unavailable"),
            _evidence("provider_healthy", "provider:healthy"),
        )
    )

    assert decision.classification.category == "unknown_inconclusive"
    assert decision.status == "inconclusive"


def test_missing_metrics_recommends_bounded_evaluation_rerun() -> None:
    decision = _service().decide(_request(_evidence("metrics_missing")))

    assert decision.classification.category == "missing_metrics"
    assert decision.recommendation.action_kind == "rerun_evaluation"
    assert decision.recommendation.registry_action == "request_recheck"


def test_recovery_json_round_trip_is_stable() -> None:
    original = _service().decide(_request(_evidence("provider_unavailable")))
    restored = RecoveryDecision.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()


def test_decision_does_not_mutate_input_evidence_or_attempt_state() -> None:
    evidence = _evidence("provider_unavailable")
    request = _request(evidence)
    original = deepcopy(request.to_dict())
    store = InMemoryRecoveryAttemptStore()

    _service(store).decide(request)

    assert request.to_dict() == original
    assert store.list_attempts() == []


def test_protocols_and_role_boundaries_are_explicit() -> None:
    store = InMemoryRecoveryAttemptStore()
    fake = FakeRecoveryActionExecutor()
    service = _service(store)
    controller = RecoveryController(
        store,
        {"retry_after_backoff": fake},
        {"retry_after_backoff"},
    )
    role = IncidentResponderRole(service, controller)

    assert isinstance(store, RecoveryAttemptStore)
    assert isinstance(fake, RecoveryActionExecutor)
    decision = role.assess(_request(_evidence("provider_unavailable")))
    assert store.list_attempts() == []
    assert role.execute_approved(decision).status == "succeeded"


def test_controller_redacts_injected_handler_failure_details() -> None:
    store = InMemoryRecoveryAttemptStore()
    decision = _service(store).decide(_request(_evidence("provider_unavailable")))
    fake = FakeRecoveryActionExecutor(failing_actions={"retry_after_backoff"})
    controller = RecoveryController(
        store,
        {"retry_after_backoff": fake},
        {"retry_after_backoff"},
    )

    result = controller.execute(decision)

    assert result.status == "failed"
    assert result.issues[0].code == "recovery_executor_failed"
    assert "deterministic fake failure" not in str(result.to_dict())


def test_unknown_and_malformed_evidence_returns_inconclusive_without_crashing() -> None:
    request = _request({"kind": "unknown_future_record", "payload": object()})
    request.evidence.append("not-a-mapping")  # type: ignore[arg-type]

    decision = _service().decide(request)

    assert decision.status == "inconclusive"
    assert decision.classification.category == "unknown_inconclusive"
    assert any(issue.code == "malformed_recovery_evidence" for issue in decision.issues)
