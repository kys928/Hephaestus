from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.schemas.judge_exit import JudgeExitAction

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/experiments/hephaestus_interface_repair_v1.json"
BUILDER = ROOT / "scripts/build_interface_repair_v1.py"
MARKER = ROOT / "configs/experiments/interface_repair_v1.launch.json"


def module():
    spec = importlib.util.spec_from_file_location("repair_pack", BUILDER)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_pack_is_frozen_and_geometry_is_exact() -> None:
    cfg = json.loads(CFG.read_text())
    m = module(); pack = m.build_pack(); m.validate(pack)
    assert m.canonical_sha256(pack) == cfg["pack"]["canonical_sha256"]
    assert cfg["frozen_diagnosis"] is True
    assert "diagnosis" not in cfg["trainable_roles"]
    for role in cfg["trainable_roles"]:
        train = pack["partitions"][role]["train"]
        assert len(train) == m.TRAIN_COUNTS[role]
        assert sum(x["kind"] == "interface" for x in train) == len(train) * 3 // 4
        assert cfg["training"]["optimizer_steps"][role] * cfg["training"]["gradient_accumulation_steps"] == len(train)
        assert len(pack["partitions"][role]["certification"]) == 128
        assert len(pack["partitions"][role]["regression"]) == 128
    assert all(len(v) == 32 for v in pack["live_partitions"].values())


def test_known_phase_ii_failure_modes_are_explicitly_repaired() -> None:
    m = module()
    # Inconclusive Diagnosis must not become a dataset branch.
    p = m.planner_case("certification", 20000, True)
    assert p["expected"]["decision"] == "collect_more_evidence"
    assert p["expected"]["action"] == "request_recheck"
    assert "branch_new_experiment" in p["forbidden_actions"]

    # Improvement/equivalence must not be upgraded to certification.
    improved = next(m.evaluator_case("certification", 20000 + i, True) for i in range(8) if m.evaluator_case("certification", 20000 + i, True)["expected"]["decision"] == "improved")
    equivalent = next(m.evaluator_case("certification", 20000 + i, True) for i in range(8) if m.evaluator_case("certification", 20000 + i, True)["expected"]["decision"] == "equivalent")
    assert improved["expected"]["action"] == "continue_from_checkpoint"
    assert equivalent["expected"]["action"] == "continue_lineage_best"
    assert "certify_candidate" in improved["forbidden_actions"]
    assert "certify_candidate" in equivalent["forbidden_actions"]

    # Strategic Judge actions are not automatically executable Controller commands.
    blocked = next(m.controller_case("certification", 20000 + i, True) for i in range(16) if m.controller_case("certification", 20000 + i, True)["upstream_output"]["action"] == "continue_lineage_best")
    assert blocked["expected"]["decision"] == "block_invalid_action"
    assert blocked["expected"]["action"] == "request_recheck"


def test_judge_gold_tracks_finite_policy_semantics() -> None:
    m = module(); policy = JudgePolicy()
    # Hard deterministic failure maps to the same reject action as the deterministic Judge policy.
    d, action, _ = m.judge_expected_from_state("evaluator", "scientific_rejection", {
        "deterministic_gate_passed": False, "evidence_complete": True, "provenance_valid": True,
        "variance_risk": "low", "monitor_outcome": "healthy",
    })
    assert d == "blocked"
    assert action == policy.decide_exit_action(False, .99, "healthy", promotion_state="rejected", has_candidate_checkpoint=True).value
    assert action == JudgeExitAction.REJECT_CHECKPOINT.value

    # Improved evidence is continuation-capable, not automatically promotion-capable.
    d, action, _ = m.judge_expected_from_state("evaluator", "improved", {
        "deterministic_gate_passed": True, "evidence_complete": True, "provenance_valid": True,
        "variance_risk": "low", "monitor_outcome": "healthy", "approval_valid": False,
        "stage_action_allowed": True, "recheck_satisfied": False,
    })
    assert d == "approved" and action == JudgeExitAction.CONTINUE_FROM_CHECKPOINT.value


def test_governance_boundary_is_frozen_before_training() -> None:
    cfg = json.loads(CFG.read_text()); marker = json.loads(MARKER.read_text())
    assert cfg["governance"]["training_approved"] is True
    assert cfg["governance"]["production_promotion_allowed"] is False
    assert cfg["governance"]["automatic_role_dispatch_allowed"] is False
    assert cfg["governance"]["source_phase_i_adapter_mutation_allowed"] is False
    assert cfg["governance"]["diagnosis_weight_mutation_allowed"] is False
    assert marker["authorized"] is True
    assert marker["authorization_phrase"] == "LAUNCH_INTERFACE_REPAIR_V1"
