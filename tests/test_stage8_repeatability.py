from __future__ import annotations

from pathlib import Path

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.policy.promotion_policy import PromotionPolicy
from hephaestus.policy.stage_policy import StagePolicy
from hephaestus.roles.evaluator import EvaluatorRole
from hephaestus.schemas.stage_profile import StageProfile
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.query import Query


def _strict_profile() -> StageProfile:
    return StagePolicy().resolve("stabilization")


def test_stage8_single_good_bundle_not_enough_for_certified_stable_when_repeatability_required() -> None:
    evaluator = EvaluatorRole()
    report = evaluator.run(
        run_id="s8-single-good",
        stage_profile=_strict_profile(),
        training_outputs={
            "intermediate_eval": {"probe_score": 0.93, "toxicity": 0.01},
            "checkpoint_candidates": [{"checkpoint_ref": "artifacts/s8-single-good/ckpt", "probe_score": 0.93}],
            "certification_evals": [{"deterministic_passed": True, "probe_score": 0.93}],
        },
    )
    policy = PromotionPolicy()
    decision = policy.decide(
        deterministic_passed=True,
        confidence=report.confidence,
        has_candidate=True,
        promotion_bundle_passed=report.promotion_bundle_passed,
        evidence_completeness=report.evidence_completeness,
        certification_readiness=report.certification_readiness,
        observed_evidence_runs=int(report.evaluation_bundle_summary["observed_evidence_runs"]),
        min_stable_evidence=2,
        min_certification_evidence=3,
        stability_confidence=report.stability_confidence,
        min_stability_confidence=float(report.evaluation_bundle_summary["min_stability_confidence"]),
        repeatability_sufficient=report.repeatability_sufficient,
        variance_risk=report.variance_risk,
    )
    assert report.certification_readiness == "certification_recheck_required"
    assert decision.promotion_state in {"promoted_best", "stable"}
    assert decision.certification_state == "certification_recheck_required"


def test_stage8_repeated_consistent_evidence_unlocks_certification() -> None:
    evaluator = EvaluatorRole()
    report = evaluator.run(
        run_id="s8-consistent",
        stage_profile=_strict_profile(),
        training_outputs={
            "intermediate_eval": {"probe_score": 0.94, "toxicity": 0.01},
            "checkpoint_candidates": [{"checkpoint_ref": "artifacts/s8-consistent/ckpt", "probe_score": 0.94}],
            "certification_evals": [
                {"deterministic_passed": True, "probe_score": 0.93},
                {"deterministic_passed": True, "probe_score": 0.94},
            ],
        },
    )
    assert report.repeatability_sufficient is True
    assert report.certification_readiness == "certification_passed"


def test_stage8_repeated_inconsistent_evidence_blocks_certification() -> None:
    evaluator = EvaluatorRole()
    report = evaluator.run(
        run_id="s8-inconsistent",
        stage_profile=_strict_profile(),
        training_outputs={
            "intermediate_eval": {"probe_score": 0.93, "toxicity": 0.01},
            "checkpoint_candidates": [{"checkpoint_ref": "artifacts/s8-inconsistent/ckpt", "probe_score": 0.93}],
            "certification_evals": [
                {"deterministic_passed": True, "probe_score": 0.93},
                {"deterministic_passed": False, "probe_score": 0.52},
            ],
        },
    )
    assert report.repeatability_blocked is True
    assert report.consistency_observed in {"mixed", "inconsistent"}
    assert report.certification_readiness in {
        "certification_blocked_by_inconsistency",
        "certification_inconclusive_due_to_variance",
    }


def test_stage8_stage_variance_sensitivity_materially_affects_outcome() -> None:
    evaluator = EvaluatorRole()
    strict = _strict_profile()
    lenient = StageProfile(
        name="lenient-repeatability",
        strictness="lenient",
        eval_pack="generic_lm",
        deterministic_gates={"max_toxicity": 0.3, "min_probe_score": 0.4},
        allowed_next_actions=["continue_lineage_best"],
        certification_profile={
            "eligibility": "bounded",
            "require_recheck": False,
            "min_consistent_runs": 1,
            "repeatability_required": True,
            "required_rechecks": 1,
            "min_repeat_consistency": 0.6,
            "variance_sensitivity": "low",
            "certification_recheck_policy": "required_if_repeatability_unmet",
        },
    )
    inputs = {
        "intermediate_eval": {"probe_score": 0.9, "toxicity": 0.01},
        "checkpoint_candidates": [{"checkpoint_ref": "artifacts/s8-var/ckpt", "probe_score": 0.9}],
        "certification_evals": [
            {"deterministic_passed": True, "probe_score": 0.8},
            {"deterministic_passed": True, "probe_score": 0.9},
        ],
    }
    strict_report = evaluator.run("s8-var-strict", strict, inputs)
    lenient_report = evaluator.run("s8-var-lenient", lenient, inputs)
    assert strict_report.variance_risk in {"high", "moderate"}
    assert lenient_report.variance_risk in {"moderate", "low"}


def test_stage8_best_stable_certified_remain_distinct_under_repeatability_logic() -> None:
    policy = PromotionPolicy()
    best = policy.decide(deterministic_passed=True, confidence=0.7, has_candidate=True)
    stable = policy.decide(
        deterministic_passed=True,
        confidence=0.92,
        has_candidate=True,
        evidence_completeness=1.0,
        certification_readiness="certification_recheck_required",
        observed_evidence_runs=3,
        min_stable_evidence=2,
        min_certification_evidence=3,
        repeatability_sufficient=False,
    )
    certified = policy.decide(
        deterministic_passed=True,
        confidence=0.98,
        has_candidate=True,
        evidence_completeness=1.0,
        certification_readiness="certification_passed",
        observed_evidence_runs=4,
        min_stable_evidence=2,
        min_certification_evidence=3,
        stability_confidence=0.98,
        min_stability_confidence=0.9,
        repeatability_sufficient=True,
    )
    assert best.promotion_state == "promoted_best"
    assert stable.promotion_state == "stable"
    assert stable.certification_state == "certification_recheck_required"
    assert certified.promotion_state == "certified_stable"


def test_stage8_query_retrieves_repeatability_history_without_policy_engine(tmp_path: Path) -> None:
    lineage = LineageStore(tmp_path)
    lineage.set_current(
        {
            "lineage_id": "lineage-main",
            "parent_lineage_id": None,
            "stage_name": "stabilization",
            "status": "active",
            "trust_level": "medium",
            "loop_index": 1,
            "latest_run_id": "s8-run",
            "best_checkpoint_ref": "artifacts/s8/query.ckpt",
            "last_stable_checkpoint_ref": None,
            "certified_stable_checkpoint_ref": None,
            "last_certification_result": "certification_recheck_required",
            "last_repeated_eval_count": 1,
            "last_consistency_score": 0.66,
            "last_variance_risk": "high",
            "certification_recheck_count": 2,
            "repeatability_sufficient": False,
            "recent_failures": [],
            "known_pathologies": ["certification_recheck_required"],
            "last_decision": "continue_from_checkpoint",
            "last_decision_id": "dec-s8-exit",
            "branch_origin_checkpoint_ref": None,
            "child_lineage_ids": [],
            "run_count": 1,
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    )
    decisions = DecisionStore(tmp_path)
    decisions.append(
        {
            "decision_id": "dec-1",
            "run_id": "r1",
            "lineage_id": "lineage-main",
            "role": "judge_exit",
            "action": "promote_checkpoint",
            "rationale": "certification_state=certification_recheck_required",
            "confidence": 0.9,
            "created_at": "2026-01-01T00:00:00+00:00",
            "metadata": {
                "checkpoint_ref": "artifacts/s8/query.ckpt",
                "certification_state": "certification_recheck_required",
                "variance_risk": "high",
                "repeated_eval_count": 1,
            },
        }
    )
    query = Query(tmp_path)
    summary = query.checkpoint_repeatability_summary("lineage-main")
    attempts = query.recent_certification_attempts_for_checkpoint("lineage-main", "artifacts/s8/query.ckpt")
    assert summary["attempt_count"] == 1
    assert summary["recent_variance_signals"] == 1
    assert attempts[0]["role"] == "judge_exit"


def test_stage8_dry_run_path_still_works_with_repeatability_logic(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(state_root=tmp_path / "dry", run_id="s8-dry", backend=DryRunBackend())
    root = Path("artifacts") / "s8-dry"
    root.mkdir(parents=True, exist_ok=True)
    (root / "processed_dataset.jsonl").write_text('{"text":"sample"}\n')
    orchestrator.run("s8-dry")
    state = LineageStore(tmp_path / "dry").get_current("lineage-main")
    assert state is not None
    assert "last_repeated_eval_count" in state
