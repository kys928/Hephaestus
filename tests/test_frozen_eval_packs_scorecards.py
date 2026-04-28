from __future__ import annotations

from pathlib import Path

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.evaluation.pack_loader import load_eval_pack
from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.policy.promotion_policy import PromotionPolicy
from hephaestus.policy.stage_policy import StagePolicy
from hephaestus.roles.evaluator import EvaluatorRole
from hephaestus.roles.judge_exit import JudgeExitRole
from hephaestus.schemas.eval_pack import EvalPack
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.scorecard import Scorecard
from hephaestus.schemas.stage_profile import StageProfile
from hephaestus.state.eval_pack_store import EvalPackStore


def test_eval_pack_normalization_integrity_levels() -> None:
    hashed = EvalPack.normalize({"pack_name": "pack-a", "content_hash": "abc123"})
    assert hashed.integrity_level == "content_hash_verified"

    ref_only = EvalPack.normalize({"pack_name": "pack-b", "source_ref": "configs/eval_packs/pack-b.yaml"})
    assert ref_only.integrity_level == "reference_only"

    inline_unhashed = EvalPack.normalize({"pack_name": "pack-c"})
    assert inline_unhashed.integrity_level == "inline_unhashed"

    insufficient = EvalPack.normalize({})
    assert insufficient.integrity_level == "insufficient"
    assert "eval_pack_identity_missing" in insufficient.warnings


def test_frozen_eval_pack_semantics_and_hash_claim_rules(tmp_path: Path) -> None:
    store = EvalPackStore(tmp_path)
    persisted = store.register({"pack_name": "persisted-pack", "frozen": False})
    assert persisted.frozen is True
    assert persisted.mutation_policy == "immutable_without_approval"

    loaded = load_eval_pack("generic_lm")
    assert loaded["frozen"] is True
    assert loaded["mutation_policy"] == "immutable_without_approval"
    assert not (loaded["eval_pack_integrity_level"] == "content_hash_verified" and not loaded["content_hash"])


def test_scorecard_gate_and_completeness_semantics() -> None:
    scorecard = Scorecard(
        scorecard_id="s-1",
        run_id="r-1",
        eval_pack_id="generic_lm",
        deterministic_passed=True,
        failed_gates=["toxicity_gate"],
        passed_gates=["probe_score_gate"],
        metrics={"probe_score": 0.9},
        thresholds={"min_probe_score": 0.8},
        gate_results={"probe_score_gate": {"passed": True}},
    ).enforce_semantics()
    assert scorecard.deterministic_passed is False
    assert "probe_score_gate" in scorecard.passed_gates

    incomplete = Scorecard(scorecard_id="s-2", run_id="r-2").enforce_semantics()
    assert incomplete.completeness_score < 1.0
    assert "deterministic_gate_results_missing" in incomplete.warnings


def test_evaluator_dry_run_smoke_includes_explicit_scorecard_and_pack_fields() -> None:
    evaluator = EvaluatorRole()
    profile = StagePolicy().resolve("stabilization")
    report = evaluator.run(
        run_id="dry-scorecard",
        stage_profile=profile,
        training_outputs={
            "intermediate_eval": {},
            "checkpoint_candidates": [{"checkpoint_ref": "artifacts/dry-scorecard/ckpt"}],
        },
    )
    assert report.eval_pack_id or report.eval_pack_integrity_level in {"reference_only", "inline_unhashed", "insufficient"}
    assert isinstance(report.deterministic_scorecard, dict)
    assert isinstance(report.deterministic_passed, bool)
    assert isinstance(report.failed_gates, list)
    assert isinstance(report.passed_gates, list)
    assert bool(report.scorecard_integrity_level)


def test_judge_promotion_compatibility_blocks_deterministic_failure_and_missing_scorecard() -> None:
    role = JudgeExitRole(judge_policy=JudgePolicy(), promotion_policy=PromotionPolicy())
    profile = StageProfile(
        name="x",
        strictness="strict",
        eval_pack="generic_lm",
        deterministic_gates={"min_probe_score": 0.5, "max_toxicity": 0.2},
    )
    bad = EvalReport(
        eval_id="e1",
        run_id="r1",
        stage_name="x",
        pack_name="generic_lm",
        checkpoint_resolution={"selected_checkpoint_ref": "artifacts/r1/ckpt"},
        regression_summary={"deterministic_passed": False},
        deterministic_scorecard={"deterministic_passed": False},
        deterministic_passed=False,
        confidence=0.99,
    )
    decision_bad = role.run("r1", "lineage-main", bad, "healthy", 0, False, profile)
    assert any("deterministic_passed=False" in reason for reason in decision_bad.reasons)

    missing_scorecard = EvalReport(
        eval_id="e2",
        run_id="r2",
        stage_name="x",
        pack_name="generic_lm",
        checkpoint_resolution={"selected_checkpoint_ref": "artifacts/r2/ckpt"},
        regression_summary={"deterministic_passed": True},
        deterministic_scorecard={},
        deterministic_passed=True,
        confidence=0.99,
    )
    decision_missing = role.run("r2", "lineage-main", missing_scorecard, "healthy", 0, False, profile)
    assert any("deterministic_passed=False" in reason for reason in decision_missing.reasons)


def test_stage_policy_exposes_eval_pack_and_gate_config() -> None:
    profile = StagePolicy().resolve("stabilization")
    assert profile.eval_pack
    assert profile.eval_pack_ref
    assert profile.deterministic_gate_config


def test_dry_run_orchestrator_path_remains_functional(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(state_root=tmp_path / "dry", run_id="scorecard-dry", backend=DryRunBackend())
    root = Path("artifacts") / "scorecard-dry"
    root.mkdir(parents=True, exist_ok=True)
    (root / "processed_dataset.jsonl").write_text('{"text":"sample"}\n')
    orchestrator.run("scorecard-dry")
