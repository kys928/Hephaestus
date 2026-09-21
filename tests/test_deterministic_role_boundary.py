from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext,
    EvidenceRecord,
    EvidenceRegistry,
    guard_role_output,
)


def _registry(**facts):
    return EvidenceRegistry(
        {
            "eval:1": EvidenceRecord(
                ref="eval:1",
                kind="evaluation",
                run_id="run-1",
                lineage_id="lin-1",
                state="verified",
                facts=facts,
            )
        }
    )


def test_unknown_evidence_reference_is_blocking():
    gate = guard_role_output(
        role="judge",
        output={"action": "continue_lineage_best", "evidence_refs": ["missing"]},
        context=DeterministicRoleContext(
            run_id="run-1",
            lineage_id="lin-1",
            stage_name="stabilization",
        ),
        evidence_registry=_registry(),
    )
    assert gate.allowed is False
    assert "unknown_evidence_ref:missing" in gate.failures


def test_wrong_lineage_evidence_is_blocking():
    registry = EvidenceRegistry(
        {
            "eval:foreign": EvidenceRecord(
                ref="eval:foreign",
                kind="evaluation",
                run_id="run-1",
                lineage_id="lin-2",
            )
        }
    )
    gate = guard_role_output(
        role="evaluator",
        output={"action": "continue_lineage_best", "evidence_refs": ["eval:foreign"]},
        context=DeterministicRoleContext(
            run_id="run-1",
            lineage_id="lin-1",
            stage_name="stabilization",
        ),
        evidence_registry=registry,
    )
    assert gate.allowed is False
    assert "wrong_lineage_evidence_ref:eval:foreign" in gate.failures


def test_hash_mismatch_is_blocking():
    registry = EvidenceRegistry(
        {
            "artifact:1": EvidenceRecord(
                ref="artifact:1",
                kind="artifact",
                run_id="run-1",
                lineage_id="lin-1",
                sha256="a" * 64,
                observed_sha256="b" * 64,
            )
        }
    )
    gate = guard_role_output(
        role="controller",
        output={"action": "continue_lineage_best", "evidence_refs": ["artifact:1"]},
        context=DeterministicRoleContext(
            run_id="run-1",
            lineage_id="lin-1",
            stage_name="stabilization",
        ),
        evidence_registry=registry,
    )
    assert gate.allowed is False
    assert "evidence_hash_mismatch:artifact:1" in gate.failures


def test_promotion_cannot_override_failed_hard_gate():
    registry = _registry(
        approval_valid=True,
        candidate_checkpoint_integrity_valid=True,
        candidate_checkpoint_lineage_match=True,
        deterministic_gate_passed=False,
        evidence_complete=True,
        provenance_valid=True,
        recheck_satisfied=True,
    )
    gate = guard_role_output(
        role="judge",
        output={"action": "promote_checkpoint", "evidence_refs": ["eval:1"]},
        context=DeterministicRoleContext(
            run_id="run-1",
            lineage_id="lin-1",
            stage_name="stabilization",
            approval_status="approved",
            stage_allowed_actions=("promote_checkpoint",),
        ),
        evidence_registry=registry,
    )
    assert gate.allowed is False
    assert "required_verified_fact_not_true:deterministic_gate_passed" in gate.failures


def test_stage_policy_is_not_left_to_model_reasoning():
    gate = guard_role_output(
        role="judge",
        output={"action": "promote_checkpoint", "evidence_refs": ["eval:1"]},
        context=DeterministicRoleContext(
            run_id="run-1",
            lineage_id="lin-1",
            stage_name="smoke_test",
            approval_status="approved",
            stage_allowed_actions=("request_recheck",),
            verified_facts={
                "approval_valid": True,
                "candidate_checkpoint_integrity_valid": True,
                "candidate_checkpoint_lineage_match": True,
                "deterministic_gate_passed": True,
                "evidence_complete": True,
                "provenance_valid": True,
                "recheck_satisfied": True,
            },
        ),
        evidence_registry=_registry(),
    )
    assert gate.allowed is False
    assert "stage_disallows_action:promote_checkpoint" in gate.failures


def test_terminal_idempotency_prevents_duplicate_mutation():
    gate = guard_role_output(
        role="controller",
        output={
            "action": "rerun_same_config",
            "evidence_refs": ["eval:1"],
            "idempotency_key": "retry-42",
        },
        context=DeterministicRoleContext(
            run_id="run-1",
            lineage_id="lin-1",
            stage_name="stabilization",
            terminal_action_keys=frozenset({"retry-42"}),
        ),
        evidence_registry=_registry(),
    )
    assert gate.allowed is False
    assert "terminal_action_already_recorded:retry-42" in gate.failures


def test_machine_verified_safe_action_passes():
    gate = guard_role_output(
        role="planner",
        output={"action": "continue_lineage_best", "evidence_refs": ["eval:1"]},
        context=DeterministicRoleContext(
            run_id="run-1",
            lineage_id="lin-1",
            stage_name="stabilization",
        ),
        evidence_registry=_registry(metrics_checksum_valid=True),
    )
    assert gate.allowed is True
    assert gate.effective_action == "continue_lineage_best"
    assert gate.validated_evidence_refs == ("eval:1",)
