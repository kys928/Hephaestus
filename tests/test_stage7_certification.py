from __future__ import annotations

import json
import shutil
from pathlib import Path

from hephaestus.backends.ardor.backend import ArdorBackend
from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.backends.local_process_backend import LocalProcessBackend
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.promotion import apply_promotion
from hephaestus.evaluation.pack_loader import load_eval_pack
from hephaestus.policy.promotion_policy import PromotionPolicy
from hephaestus.policy.stage_policy import StagePolicy
from hephaestus.roles.evaluator import EvaluatorRole
from hephaestus.schemas.stage_profile import StageProfile
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.query import Query


_FIXTURE_RUNNER = "tests/fixtures/fake_ardor_runner.py"


def _config_dir_with_ardor_runner(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs-with-runner"
    shutil.copytree("configs", config_dir)
    ardor_cfg = json.loads((config_dir / "backends" / "ardor.yaml").read_text())
    ardor_cfg["local_runner_path"] = _FIXTURE_RUNNER
    (config_dir / "backends" / "ardor.yaml").write_text(json.dumps(ardor_cfg, indent=2))
    return config_dir


def _stage_profile(name: str = "stabilization") -> StageProfile:
    return StagePolicy().resolve(name)


def test_stage7_richer_eval_pack_loads_and_is_stage_aware() -> None:
    pack = load_eval_pack("continuation_repair")
    assert "certification_bundle" in pack
    assert "minimum_evidence" in pack
    assert pack["stage_tolerances"]["strict"]["min_confidence_certified"] > pack["stage_tolerances"]["strict"]["min_confidence_stable"]


def test_stage7_evaluator_summarizes_multi_part_eval_bundle() -> None:
    evaluator = EvaluatorRole()
    report = evaluator.run(
        run_id="s7-eval-1",
        stage_profile=_stage_profile("stabilization"),
        training_outputs={
            "intermediate_eval": {"probe_score": 0.9, "toxicity": 0.01},
            "checkpoint_candidates": [{"checkpoint_ref": "artifacts/s7-eval-1/ckpt-1", "probe_score": 0.9}],
            "certification_evals": [
                {"deterministic_passed": True},
                {"deterministic_passed": True},
            ],
        },
    )
    assert report.evaluation_bundle_summary["observed_evidence_runs"] == 3
    assert report.evidence_completeness == 1.0
    assert report.certification_readiness in {"certification_passed", "certification_inconclusive", "certification_recheck_required"}


def test_stage7_certification_blocked_when_deterministic_regression_fails() -> None:
    policy = PromotionPolicy()
    decision = policy.decide(
        deterministic_passed=False,
        confidence=0.99,
        has_candidate=True,
        evidence_completeness=1.0,
        certification_readiness="certification_passed",
    )
    assert decision.promotion_state == "rejected"
    assert decision.certification_state == "certification_blocked_by_regression"


def test_stage7_best_stable_and_certified_are_distinct() -> None:
    policy = PromotionPolicy()
    best = policy.decide(deterministic_passed=True, confidence=0.7, has_candidate=True, evidence_completeness=0.4)
    stable = policy.decide(
        deterministic_passed=True,
        confidence=0.9,
        has_candidate=True,
        evidence_completeness=1.0,
        certification_readiness="certification_inconclusive",
        observed_evidence_runs=2,
        min_stable_evidence=2,
    )
    certified = policy.decide(
        deterministic_passed=True,
        confidence=0.98,
        has_candidate=True,
        evidence_completeness=1.0,
        certification_readiness="certification_passed",
        observed_evidence_runs=3,
        min_stable_evidence=2,
        min_certification_evidence=3,
    )
    assert best.promotion_state == "promoted_best"
    assert stable.promotion_state == "stable"
    assert certified.promotion_state == "certified_stable"


def test_stage7_inconclusive_or_recheck_evidence_blocks_certification() -> None:
    policy = PromotionPolicy()
    inconclusive = policy.decide(
        deterministic_passed=True,
        confidence=0.97,
        has_candidate=True,
        evidence_completeness=1.0,
        certification_readiness="certification_inconclusive",
        observed_evidence_runs=3,
        min_stable_evidence=2,
        min_certification_evidence=3,
    )
    recheck = policy.decide(
        deterministic_passed=True,
        confidence=0.97,
        has_candidate=True,
        evidence_completeness=1.0,
        certification_readiness="certification_passed",
        recheck_recommended=True,
        observed_evidence_runs=3,
        min_stable_evidence=2,
        min_certification_evidence=3,
    )
    assert inconclusive.promotion_state == "stable"
    assert recheck.promotion_state == "stable"
    assert recheck.certification_state == "certification_recheck_required"


def test_stage7_stage_certification_eligibility_is_enforced() -> None:
    policy = PromotionPolicy()
    decision = policy.decide(
        deterministic_passed=True,
        confidence=0.98,
        has_candidate=True,
        promotion_bundle_passed=True,
        evidence_completeness=1.0,
        certification_readiness="certification_passed",
        stage_certification_eligibility="disabled",
        observed_evidence_runs=4,
        min_promotion_evidence=1,
        min_stable_evidence=2,
        min_certification_evidence=3,
        stability_confidence=0.98,
        min_stability_confidence=0.9,
    )
    assert decision.promotion_state == "stable"
    assert decision.certification_state == "certification_not_eligible"


def test_stage7_evaluator_applies_stage_certification_profile_controls() -> None:
    evaluator = EvaluatorRole()
    report = evaluator.run(
        run_id="s7-stage-profile",
        stage_profile=_stage_profile("stabilization"),
        training_outputs={
            "intermediate_eval": {"probe_score": 0.91, "toxicity": 0.01},
            "checkpoint_candidates": [{"checkpoint_ref": "artifacts/s7-stage-profile/ckpt", "probe_score": 0.91}],
            "certification_evals": [{"deterministic_passed": True}],
        },
    )
    assert report.recheck_recommended is True
    assert report.evaluation_bundle_summary["effective_min_consistent_runs"] == 2
    assert report.observed_consistent_runs == 2


def test_stage7_evaluator_marks_not_eligible_when_stage_disables_certification() -> None:
    evaluator = EvaluatorRole()
    stage_profile = StageProfile(
        name="disabled-cert-stage",
        strictness="strict",
        eval_pack="continuation_repair",
        deterministic_gates={"max_toxicity": 0.06, "min_probe_score": 0.74},
        allowed_next_actions=["promote_checkpoint"],
        certification_profile={"eligibility": "disabled", "require_recheck": False, "min_consistent_runs": 1},
    )
    report = evaluator.run(
        run_id="s7-stage-disabled",
        stage_profile=stage_profile,
        training_outputs={
            "intermediate_eval": {"probe_score": 0.92, "toxicity": 0.01},
            "checkpoint_candidates": [{"checkpoint_ref": "artifacts/s7-stage-disabled/ckpt", "probe_score": 0.92}],
            "certification_evals": [{"deterministic_passed": True}, {"deterministic_passed": True}],
        },
    )
    assert report.certification_readiness == "certification_not_eligible"


def test_stage7_stage_require_recheck_and_min_consistent_runs_are_enforced() -> None:
    policy = PromotionPolicy()
    decision = policy.decide(
        deterministic_passed=True,
        confidence=0.98,
        has_candidate=True,
        promotion_bundle_passed=True,
        evidence_completeness=1.0,
        certification_readiness="certification_passed",
        stage_require_recheck=True,
        stage_min_consistent_runs=3,
        observed_consistent_runs=2,
        observed_evidence_runs=4,
        min_promotion_evidence=1,
        min_stable_evidence=2,
        min_certification_evidence=3,
        stability_confidence=0.98,
        min_stability_confidence=0.9,
    )
    assert decision.promotion_state == "stable"
    assert decision.certification_state == "certification_recheck_required"


def test_stage7_promotion_runs_blocks_promotion_when_unmet() -> None:
    policy = PromotionPolicy()
    decision = policy.decide(
        deterministic_passed=True,
        confidence=0.9,
        has_candidate=True,
        promotion_bundle_passed=True,
        evidence_completeness=1.0,
        observed_evidence_runs=1,
        min_promotion_evidence=2,
        min_stable_evidence=2,
    )
    assert decision.promotion_state == "candidate_best"


def test_stage7_min_stability_confidence_blocks_certification() -> None:
    policy = PromotionPolicy()
    decision = policy.decide(
        deterministic_passed=True,
        confidence=0.98,
        has_candidate=True,
        promotion_bundle_passed=True,
        evidence_completeness=1.0,
        certification_readiness="certification_passed",
        observed_evidence_runs=4,
        min_promotion_evidence=1,
        min_stable_evidence=2,
        min_certification_evidence=3,
        stability_confidence=0.82,
        min_stability_confidence=0.9,
    )
    assert decision.promotion_state == "stable"
    assert decision.certification_state == "certification_inconclusive"


def test_stage7_promotion_bundle_failure_blocks_advancement() -> None:
    policy = PromotionPolicy()
    decision = policy.decide(
        deterministic_passed=True,
        confidence=0.99,
        has_candidate=True,
        promotion_bundle_passed=False,
        evidence_completeness=1.0,
        certification_readiness="certification_passed",
        observed_evidence_runs=4,
        min_promotion_evidence=1,
        min_stable_evidence=2,
        min_certification_evidence=3,
        stability_confidence=0.99,
        min_stability_confidence=0.9,
    )
    assert decision.promotion_state == "candidate_best"
    assert decision.certification_state == "certification_blocked_by_regression"


def test_stage7_lineage_tracks_certified_stable_separately() -> None:
    transition = apply_promotion(
        lineage_state={
            "best_checkpoint_ref": "artifacts/old/best.ckpt",
            "last_stable_checkpoint_ref": "artifacts/old/stable.ckpt",
            "certified_stable_checkpoint_ref": None,
        },
        candidate_checkpoint_ref="artifacts/new/cert.ckpt",
        promotion_state="certified_stable",
        certification_state="certification_passed",
        deterministic_passed=True,
        confidence=0.99,
        stable_confidence_threshold=0.85,
    )
    assert transition.best_checkpoint_ref == "artifacts/new/cert.ckpt"
    assert transition.last_stable_checkpoint_ref == "artifacts/new/cert.ckpt"
    assert transition.certified_stable_checkpoint_ref == "artifacts/new/cert.ckpt"


def test_stage7_query_support_for_certification_fields(tmp_path: Path) -> None:
    store = LineageStore(tmp_path)
    store.set_current(
        {
            "lineage_id": "lineage-main",
            "parent_lineage_id": None,
            "stage_name": "stabilization",
            "status": "active",
            "trust_level": "high",
            "loop_index": 3,
            "latest_run_id": "s7-run-3",
            "best_checkpoint_ref": "artifacts/s7/best.ckpt",
            "last_stable_checkpoint_ref": "artifacts/s7/stable.ckpt",
            "certified_stable_checkpoint_ref": "artifacts/s7/cert.ckpt",
            "last_certification_result": "certification_passed",
            "recent_failures": [],
            "known_pathologies": [],
            "last_decision": "promote_checkpoint",
            "last_decision_id": "dec-s7-run-3-exit",
            "branch_origin_checkpoint_ref": None,
            "child_lineage_ids": [],
            "run_count": 3,
            "updated_at": "2026-01-01T00:00:00+00:00"
        }
    )
    query = Query(tmp_path)
    assert query.best_checkpoint("lineage-main") == "artifacts/s7/best.ckpt"
    assert query.last_stable_checkpoint("lineage-main") == "artifacts/s7/stable.ckpt"
    assert query.certified_stable_checkpoint("lineage-main") == "artifacts/s7/cert.ckpt"
    assert query.last_certification_decision("lineage-main") == "certification_passed"


def test_stage7_dry_local_ardor_paths_still_work(tmp_path: Path) -> None:
    dry = build_orchestrator(state_root=tmp_path / "dry", run_id="s7-dry", backend=DryRunBackend())
    local = build_orchestrator(state_root=tmp_path / "local", run_id="s7-local", backend=LocalProcessBackend())
    ardor = build_orchestrator(
        state_root=tmp_path / "ardor",
        run_id="s7-ardor",
        backend=ArdorBackend(config_dir=_config_dir_with_ardor_runner(tmp_path)),
    )

    for run_id in ["s7-dry", "s7-local", "s7-ardor"]:
        root = Path("artifacts") / run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "processed_dataset.jsonl").write_text('{"text":"sample"}\n')

    dry.run("s7-dry")
    local.run("s7-local")
    ardor.run("s7-ardor")

    assert LineageStore(tmp_path / "dry").get_current("lineage-main") is not None
    assert LineageStore(tmp_path / "local").get_current("lineage-main") is not None
    assert LineageStore(tmp_path / "ardor").get_current("lineage-main") is not None
