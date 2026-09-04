from hephaestus.control.semantic_judge import SemanticComparisonJudgeAdapter
from hephaestus.schemas.experiment_contract import ExperimentComparison
from hephaestus.schemas.judge_exit import JudgeExitAction


def _comparison(outcome: str, *, gate: str = "passed", confidence: float = 0.7, recommendation: str = "collect_more_evidence") -> ExperimentComparison:
    return ExperimentComparison(
        comparison_id="comparison-1",
        experiment_id="experiment-1",
        baseline_run_id="baseline",
        candidate_run_ids=["candidate"],
        primary_outcome=outcome,
        deterministic_gate_status=gate,
        variance_risk="low",
        recommendation=recommendation,
        confidence=confidence,
        effect_summary={
            "missing_evidence": [],
            "repeatability": {
                "candidate_run_count": 1,
                "direction_consistency": 1.0,
            },
        },
    )


def _decide(comparison: ExperimentComparison):
    return SemanticComparisonJudgeAdapter().decide(
        comparison,
        run_id="candidate",
        lineage_id="lineage-1",
        candidate_checkpoint_ref="checkpoint://candidate",
    )


def test_regressed_semantic_evidence_is_rejected_even_without_hard_failure() -> None:
    decision = _decide(_comparison("regressed", recommendation="reject_candidate_evidence"))
    assert decision.next_action == JudgeExitAction.REJECT_CHECKPOINT


def test_equivalent_evidence_retains_baseline() -> None:
    decision = _decide(_comparison("equivalent_within_evidence", recommendation="retain_baseline"))
    assert decision.next_action == JudgeExitAction.CONTINUE_LINEAGE_BEST


def test_improved_evidence_with_pending_human_review_cannot_promote() -> None:
    decision = _decide(
        _comparison(
            "improved",
            confidence=0.75,
            recommendation="consider_candidate_after_human_review",
        )
    )
    assert decision.next_action == JudgeExitAction.CONTINUE_FROM_CHECKPOINT
    assert any(reason == "human_review_pending=True" for reason in decision.reasons)


def test_hard_deterministic_failure_is_rejected() -> None:
    decision = _decide(
        _comparison(
            "mixed",
            gate="failed",
            recommendation="human_review_required",
        )
    )
    assert decision.next_action == JudgeExitAction.REJECT_CHECKPOINT
