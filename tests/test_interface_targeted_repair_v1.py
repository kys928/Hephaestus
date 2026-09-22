from __future__ import annotations
import importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/experiments/hephaestus_interface_targeted_repair_v1.json"
MARKER=ROOT/"configs/experiments/interface_targeted_repair_v1.launch.json"
REPAIR=ROOT/"scripts/build_interface_repair_v1.py"
IIB=ROOT/"scripts/build_interface_mastery_v1b.py"

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);assert s and s.loader
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_repair_training_uses_only_repair_train_and_dev() -> None:
    cfg=json.loads(CFG.read_text());repair=load(REPAIR,"repair_training_pack");pack=repair.build_pack();repair.validate(pack)
    assert cfg["training"]["roles_in_order"]==["controller","evaluator","judge"]
    for role in cfg["training"]["roles_in_order"]:
        rows=pack["samples"][role]
        assert len([r for r in rows if r["split"]=="train"])==32
        assert len([r for r in rows if r["split"]=="dev"])==8
    assert cfg["repair_pack"]["phase_ii_v1_training_use"] is False
    assert cfg["repair_pack"]["phase_ii_b_training_use"] is False
    assert cfg["governance"]["phase_ii_b_access_during_training_allowed"] is False

def test_phase_ii_b_ids_are_not_repair_ids() -> None:
    repair=load(REPAIR,"repair_ids");iib=load(IIB,"iib_ids")
    rp=repair.build_pack();bp=iib.build_pack()
    rids={r["sample_id"] for rows in rp["samples"].values() for r in rows}
    bids={r["case_id"] for rows in bp["partitions"].values() for r in rows}
    assert rids.isdisjoint(bids)

def test_repair_launch_stays_blocked_until_ab_is_frozen() -> None:
    cfg=json.loads(CFG.read_text());marker=json.loads(MARKER.read_text())
    if cfg["normalization_gate"]["required_ab_run_id"] is None:
        assert marker["authorized"] is False
    assert cfg["governance"]["production_promotion_allowed"] is False
    assert cfg["governance"]["automatic_role_dispatch_allowed"] is False
    assert cfg["governance"]["phase_i_registry_mutation_allowed"] is False
