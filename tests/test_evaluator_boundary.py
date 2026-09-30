from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext,
    EvidenceRecord,
    EvidenceRegistry,
    guard_role_output,
)
from hephaestus.control.evaluator_boundary import (
    DECISION_TO_ACTION,
    action_for_evaluator_decision,
    project_evaluator_action,
)


def _facts(**overrides):
    base = {
        "deterministic_gate_passed": True,
        "evidence_complete": True,
        "provenance_valid": True,
        "recheck_satisfied": True,
    }
    base.update(overrides)
    return base


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


def test_decision_table_is_finite_and_matches_hephaestus_policy():
    assert DECISION_TO_ACTION == {
        "scientific_rejection": "reject_candidate",
        "incomplete_evidence": "request_recheck",
        "improved": "continue_from_checkpoint",
        "regressed": "continue_lineage_best",
        "equivalent": "continue_lineage_best",
        "inconclusive": "hold_candidate",
        "recheck_required": "request_recheck",
        "certification_ready": "certify_candidate",
    }
    assert action_for_evaluator_decision("equivalent") == "continue_lineage_best"
    assert action_for_evaluator_decision("unknown") is None


def test_equivalent_can_never_be_projected_to_certify_candidate():
    projection = project_evaluator_action(
        "equivalent",
        _facts(),
        model_proposed_action="certify_candidate",
    )
    assert projection.blocked is False
    assert projection.effective_action == "continue_lineage_best"
    assert projection.model_action_agrees is False
    assert projection.machine_override is False


def test_improved_projects_to_continue_from_checkpoint_even_if_model_is_conservative():
    projection = project_evaluator_action(
        "improved",
        _facts(),
        model_proposed_action="continue_lineage_best",
    )
    assert projection.effective_action == "continue_from_checkpoint"
    assert projection.model_action_agrees is False


def test_failed_hard_gate_overrides_optimistic_scientific_decision():
    projection = project_evaluator_action(
        "improved",
        _facts(deterministic_gate_passed=False),
        model_proposed_action="continue_from_checkpoint",
    )
    assert projection.effective_action == "reject_candidate"
    assert projection.machine_override is True
    assert "verified_hard_gate_failure_forces_rejection" in projection.reasons


def test_certification_ready_requires_every_machine_fact_to_be_true():
    projection = project_evaluator_action(
        "certification_ready",
        _facts(recheck_satisfied=False),
        model_proposed_action="certify_candidate",
    )
    assert projection.effective_action == "request_recheck"
    assert projection.machine_override is True
    assert "certification_fact_not_true:recheck_satisfied" in projection.reasons


def test_unknown_scientific_decision_is_blocked():
    projection = project_evaluator_action("looks_good", _facts())
    assert projection.blocked is True
    assert projection.effective_action is None


def test_role_boundary_executes_projection_not_model_evaluator_action():
    facts = _facts()
    gate = guard_role_output(
        role="evaluator",
        output={
            "decision": "equivalent",
            "action": "certify_candidate",
            "evidence_refs": ["eval:1"],
        },
        context=DeterministicRoleContext(
            run_id="run-1",
            lineage_id="lin-1",
            stage_name="evaluation",
            stage_allowed_actions=("continue_lineage_best",),
            verified_facts=facts,
        ),
        evidence_registry=_registry(),
    )
    assert gate.allowed is True
    assert gate.requested_action == "certify_candidate"
    assert gate.effective_action == "continue_lineage_best"
    assert gate.policy_source == "deterministic_evaluator_boundary"
    assert gate.model_action_agrees is False


def test_role_boundary_blocks_projection_when_verified_evidence_is_foreign():
    gate = guard_role_output(
        role="evaluator",
        output={
            "decision": "improved",
            "action": "continue_from_checkpoint",
            "evidence_refs": ["foreign"],
        },
        context=DeterministicRoleContext(
            run_id="run-1",
            lineage_id="lin-1",
            stage_name="evaluation",
            verified_facts=_facts(),
        ),
        evidence_registry=EvidenceRegistry(
            {
                "foreign": EvidenceRecord(
                    ref="foreign",
                    kind="evaluation",
                    run_id="run-1",
                    lineage_id="lin-2",
                )
            }
        ),
    )
    assert gate.allowed is False
    assert gate.effective_action is None
    assert "wrong_lineage_evidence_ref:foreign" in gate.failures
