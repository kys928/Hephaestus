from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from hephaestus.evaluation.handoff_normalization import normalize_handoff, render_normalized_handoff

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "scripts/build_interface_mastery_v1.py"
REPAIR = ROOT / "scripts/build_interface_repair_v1.py"
IIB = ROOT / "scripts/build_interface_mastery_v1b.py"
REPAIR_CFG = ROOT / "configs/experiments/hephaestus_interface_repair_v1.json"
IIB_CFG = ROOT / "configs/experiments/hephaestus_interface_mastery_v1b.json"
AB_CFG = ROOT / "configs/experiments/hephaestus_interface_normalization_ab_v1.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_handoff_normalization_is_deterministic_and_non_authoritative() -> None:
    raw = '{"decision":"approved","action":"promote_checkpoint","primary_variable":"certification_state","confidence":0.99,"evidence_refs":["E1"],"uncertainties":[],"rationale":"ok"}'
    envelope = normalize_handoff(source_role="judge", raw_output=raw)
    assert envelope["authority"] == "untrusted_model_output"
    assert envelope["schema_copy_allowed"] is False
    assert envelope["source_role"] == "judge"
    assert envelope["payload"]["primary_variable"] == "certification_state"
    assert render_normalized_handoff(source_role="judge", raw_output=raw) == render_normalized_handoff(source_role="judge", raw_output=raw)


def test_repair_data_is_separate_valid_and_role_targeted() -> None:
    v1, repair = load(V1, "v1_for_repair_test"), load(REPAIR, "repair_pack")
    v1_pack, repair_pack = v1.build_pack(), repair.build_pack()
    v1.validate(v1_pack); repair.validate(repair_pack)
    v1_ids = {c["case_id"] for rows in v1_pack["partitions"].values() for c in rows}
    repair_ids = {r["sample_id"] for rows in repair_pack["samples"].values() for r in rows}
    assert len(repair_ids) == 96
    assert not (v1_ids & repair_ids)
    assert set(repair_pack["samples"]) == {"controller", "evaluator", "judge"}
    cfg = json.loads(REPAIR_CFG.read_text())
    assert cfg["leakage_policy"]["phase_ii_v1_training_use"] is False
    assert cfg["leakage_policy"]["phase_ii_b_training_use"] is False


def test_phase_ii_b_is_fresh_frozen_and_not_training_data() -> None:
    v1, repair, iib = load(V1, "v1_for_iib_test"), load(REPAIR, "repair_for_iib_test"), load(IIB, "iib_pack")
    v1_pack, repair_pack, iib_pack = v1.build_pack(), repair.build_pack(), iib.build_pack()
    iib.validate(iib_pack)
    v1_ids = {c["case_id"] for rows in v1_pack["partitions"].values() for c in rows}
    repair_ids = {r["sample_id"] for rows in repair_pack["samples"].values() for r in rows}
    iib_ids = {c["case_id"] for rows in iib_pack["partitions"].values() for c in rows}
    assert len(iib_ids) == 32
    assert not (iib_ids & v1_ids)
    assert not (iib_ids & repair_ids)
    cfg = json.loads(IIB_CFG.read_text())
    assert cfg["pack"]["frozen"] is True
    assert cfg["governance"]["training_use_allowed"] is False
    assert cfg["governance"]["must_remain_unseen_until_post_repair_recertification"] is True


def test_normalization_ab_is_inference_only_and_paired() -> None:
    cfg = json.loads(AB_CFG.read_text())
    assert cfg["pairing"]["generate_producer_once_per_case"] is True
    assert cfg["pairing"]["reuse_exact_producer_output_across_arms"] is True
    assert cfg["pairing"]["same_consumer_seed_across_arms"] is True
    assert cfg["governance"]["weights_mutation_allowed"] is False
    assert cfg["governance"]["targeted_repair_training_allowed"] is False
    assert cfg["source_phase_ii"]["required_pack_sha256"] == "2d712d7bdd9dc51dfefc677389aca97ce80e64c4825df38243bc0d6cfe13bad0"
