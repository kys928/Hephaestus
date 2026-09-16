from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_distance_to_role_v1 as dtr


def _protocols():
    protocol = json.loads((ROOT / "configs/experiments/hephaestus_distance_to_role_v1.json").read_text())
    topology = json.loads((ROOT / "configs/eval_packs/hephaestus_cognitive_topology_v1.json").read_text())
    semantic = json.loads((ROOT / "configs/eval_packs/semantic_behavior_v1.yaml").read_text())
    return protocol, topology, semantic


def test_distance_protocol_is_bounded_and_non_mutating() -> None:
    protocol, topology, semantic = _protocols()
    dtr.validate_protocol(protocol, topology, semantic)
    assert protocol["training"]["dose_optimizer_steps"] == [3, 6, 12, 24, 36, 48]
    assert protocol["training"]["expected_optimizer_steps_per_epoch"] == 12
    assert protocol["training"]["max_epochs"] == 4
    assert protocol["role_screening"]["quality_thresholds_100"] == [60, 70, 80, 90]
    assert protocol["governance"]["training_is_experimentally_approved"] is True
    assert protocol["governance"]["promotion_allowed"] is False
    assert protocol["governance"]["lineage_mutation_allowed"] is False
    assert protocol["execution"]["persistent_model_cache_allowed"] is False


def test_shards_exactly_partition_frozen_candidates() -> None:
    protocol, topology, _ = _protocols()
    expected = [row["model_id"] for row in topology["candidates"]]
    shards = protocol["execution"]["model_shards"]
    assert [*shards["olmo-granite"], *shards["qwen-ministral"]] == expected


def test_semantic_task_identity_stays_frozen() -> None:
    _, _, semantic = _protocols()
    assert [row["task_id"] for row in dtr.semantic_tasks(semantic)] == [
        "instruction_triplet", "planet_fact", "observatory_continuation",
        "structured_planet_answer", "anti_repetition", "brief_termination"
    ]


def _role_summary(quality: float, *, schema: float = 1.0, hallucination: float = 0.0):
    return {"quality_100": quality, "schema_compliance": schema, "hallucination_rate": hallucination}


def test_safety_separates_interface_and_semantic_regression() -> None:
    protocol, _, _ = _protocols()
    baseline = {"mean_score": 0.80, "hard_failures": ["existing"]}
    same = {"mean_score": 0.80, "hard_failures": ["existing"]}
    state = dtr.safe_state(_role_summary(70), same, baseline, protocol)
    assert state["interface_ok"] is True
    assert state["strict_safe"] is True
    assert state["bounded_safe"] is True

    mild = {"mean_score": 0.78, "hard_failures": ["existing"]}
    state = dtr.safe_state(_role_summary(75), mild, baseline, protocol)
    assert state["strict_safe"] is False
    assert state["bounded_safe"] is True

    hard = {"mean_score": 0.90, "hard_failures": ["existing", "new"]}
    state = dtr.safe_state(_role_summary(90), hard, baseline, protocol)
    assert state["strict_safe"] is False
    assert state["bounded_safe"] is False
    assert state["new_hard_failures"] == ["new"]


def test_threshold_distance_uses_first_safe_dose_not_peak() -> None:
    points = []
    for steps, quality, strict, bounded in [
        (0, 55, True, True), (3, 61, True, True), (6, 72, False, True),
        (12, 81, False, False), (24, 92, True, True), (36, 94, True, True), (48, 90, True, True)
    ]:
        points.append({
            "optimizer_steps": steps,
            "role": _role_summary(quality),
            "safety": {"interface_ok": True, "strict_safe": strict, "bounded_safe": bounded, "semantic_score_delta": 0.0},
            "training": {"supervised_tokens_cumulative": steps * 100}
        })
    distances = dtr.threshold_distances(points, [60, 70, 80, 90])
    assert distances["raw"]["60"]["optimizer_steps"] == 3
    assert distances["raw"]["70"]["optimizer_steps"] == 6
    assert distances["strict_safe"]["70"]["optimizer_steps"] == 24
    assert distances["bounded_safe"]["70"]["optimizer_steps"] == 6
    assert distances["strict_safe"]["90"]["optimizer_steps"] == 24


def test_threshold_is_unreached_when_machine_interface_fails() -> None:
    points = [{
        "optimizer_steps": 0,
        "role": _role_summary(95, schema=0.0, hallucination=1.0),
        "safety": {"interface_ok": False, "strict_safe": False, "bounded_safe": False, "semantic_score_delta": 0.0},
        "training": {"supervised_tokens_cumulative": 0}
    }]
    distances = dtr.threshold_distances(points, [90])
    assert distances["raw"]["90"] is None
    assert distances["strict_safe"]["90"] is None


def test_selection_prefers_earlier_dose_on_equal_quality() -> None:
    points = [
        {"optimizer_steps": 0, "role": _role_summary(60), "safety": {"interface_ok": True, "strict_safe": True, "bounded_safe": True}},
        {"optimizer_steps": 3, "role": _role_summary(80), "safety": {"interface_ok": True, "strict_safe": True, "bounded_safe": True}},
        {"optimizer_steps": 6, "role": _role_summary(80), "safety": {"interface_ok": True, "strict_safe": True, "bounded_safe": True}}
    ]
    selected = dtr.select_points(points)
    assert selected == {"raw_peak_steps": 3, "strict_safe_peak_steps": 3, "bounded_safe_peak_steps": 3}


def test_selection_marks_missing_safe_frontier_explicitly() -> None:
    points = [{"optimizer_steps": 0, "role": _role_summary(95, schema=0.0, hallucination=1.0), "safety": {"interface_ok": False, "strict_safe": False, "bounded_safe": False}}]
    selected = dtr.select_points(points)
    assert selected == {"raw_peak_steps": None, "strict_safe_peak_steps": None, "bounded_safe_peak_steps": None}
