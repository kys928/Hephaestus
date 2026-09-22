from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from hephaestus.evaluation.interface_mastery import score_interface_case, summarize_scorecards
from hephaestus.providers.models.role_stack import load_certified_role_model_stack

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/experiments/hephaestus_interface_mastery_v1.json"
MARKER = ROOT / "configs/experiments/interface_mastery_v1.launch.json"
STACK = ROOT / "configs/models/hephaestus_role_model_stack_v1.json"
BUILDER = ROOT / "scripts/build_interface_mastery_v1.py"


def _builder_module():
    spec = importlib.util.spec_from_file_location("interface_mastery_pack", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _perfect_output(expected: dict[str, object], refs: list[str]) -> str:
    return json.dumps(
        {
            "decision": expected["decision"],
            "action": expected["action"],
            "primary_variable": expected["primary_variable"],
            "confidence": (float(expected["confidence_min"]) + float(expected["confidence_max"])) / 2,
            "evidence_refs": refs,
            "uncertainties": [],
            "rationale": "The verified evidence supports exactly this bounded interface decision.",
        },
        separators=(",", ":"),
    )


def test_phase_ii_pack_is_frozen_complete_and_role_covering() -> None:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    module = _builder_module()
    pack = module.build_pack()
    module.validate(pack)
    assert module.canonical_sha256(pack) == cfg["pack"]["canonical_sha256"]
    assert sum(len(rows) for rows in pack["partitions"].values()) == 32
    roles = {
        role
        for rows in pack["partitions"].values()
        for case in rows
        for role in (case["producer_role"], case["consumer_role"])
    }
    assert roles == {"controller", "diagnosis", "planner", "evaluator", "judge"}


def test_phase_ii_uses_certified_phase_i_stack_without_enabling_dispatch() -> None:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    stack = load_certified_role_model_stack(STACK)
    assert stack.source_run_id == cfg["role_stack"]["required_source_run_id"]
    assert stack.production_certified is True
    assert stack.runtime_registry_ready is True
    assert stack.automatic_role_dispatch_enabled is False
    assert cfg["governance"]["automatic_role_dispatch_allowed"] is False
    assert cfg["governance"]["weights_mutation_allowed"] is False
    assert cfg["governance"]["production_promotion_allowed"] is False
    assert cfg["governance"]["targeted_repair_training_allowed"] is False


def test_phase_ii_launch_marker_is_frozen_and_well_formed() -> None:
    marker = json.loads(MARKER.read_text(encoding="utf-8"))
    assert marker["protocol_id"] == "hephaestus_interface_mastery_v1"
    assert marker["authorization_phrase"] == "LAUNCH_PHASE_II_INTERFACE_MASTERY_V1"
    if marker["authorized"] is True:
        assert marker["authorized_by"]
        assert marker["authorized_at"]
    else:
        assert marker["authorized"] is False
        assert marker["authorized_by"] is None
        assert marker["authorized_at"] is None


def test_interface_scorecard_accepts_exact_grounded_handoff() -> None:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    module = _builder_module()
    pack = module.build_pack()
    case = pack["partitions"]["diagnosis_to_planner"][0]
    producer = _perfect_output(case["producer_expected"], case["allowed_evidence_refs"])
    consumer = _perfect_output(case["consumer_expected"], case["allowed_evidence_refs"])
    score = score_interface_case(
        case=case,
        producer_raw=producer,
        consumer_raw=consumer,
        producer_vocabulary=pack["contract_vocabulary"][case["producer_role"]],
        consumer_vocabulary=pack["contract_vocabulary"][case["consumer_role"]],
    )
    assert score.quality_100 == 100.0
    assert not score.failures
    summary = summarize_scorecards([score], cfg["certification"], protocol_id=cfg["protocol_id"])
    assert summary["overall_quality_100"] == 100.0


def test_interface_scorecard_rejects_hallucinated_evidence_reference() -> None:
    module = _builder_module()
    pack = module.build_pack()
    case = pack["partitions"]["evaluator_to_judge"][0]
    producer = _perfect_output(case["producer_expected"], case["allowed_evidence_refs"])
    consumer_payload = json.loads(_perfect_output(case["consumer_expected"], case["allowed_evidence_refs"]))
    consumer_payload["evidence_refs"].append("E-II-NOT-REAL")
    score = score_interface_case(
        case=case,
        producer_raw=producer,
        consumer_raw=json.dumps(consumer_payload),
        producer_vocabulary=pack["contract_vocabulary"][case["producer_role"]],
        consumer_vocabulary=pack["contract_vocabulary"][case["consumer_role"]],
    )
    assert score.evidence_grounded is False
    assert "E-II-NOT-REAL" in score.hallucinated_evidence_refs
    assert "hallucinated_evidence_ref" in score.failures
