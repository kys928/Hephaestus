import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load():
    p=ROOT/"scripts/build_planner_judge_bakeoff_v1.py"
    s=importlib.util.spec_from_file_location("pj",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_pack_counts_and_validation():
    m=load();p=m.build_pack();m.validate(p)
    assert len(p["partitions"]["planner"]["zero_shot"])==32
    assert len(p["partitions"]["planner"]["micro_lora_train"])==80
    assert len(p["partitions"]["planner"]["held_out"])==48
    assert len(p["partitions"]["judge"]["zero_shot"])==32
    assert len(p["partitions"]["judge"]["micro_lora_train"])==80
    assert len(p["partitions"]["judge"]["held_out"])==48
def test_every_root_is_represented_in_every_split():
    m=load();p=m.build_pack()
    for role in ("planner","judge"):
        roots=set(m.ROOT_SKILLS[role])
        for split in m.SPLIT_COUNTS:
            assert {x["root_case_id"] for x in p["partitions"][role][split]}==roots
def test_split_ids_are_disjoint():
    m=load();p=m.build_pack()
    for role in ("planner","judge"):
        sets=[{x["case_id"] for x in p["partitions"][role][s]} for s in m.SPLIT_COUNTS]
        assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
def test_planner_contains_one_variable_and_approval_cases():
    m=load();p=m.build_pack();rows=p["partitions"]["planner"]["held_out"]
    p2=next(x for x in rows if x["root_case_id"]=="P2")
    p8=next(x for x in rows if x["root_case_id"]=="P8")
    assert p2["expected"]["primary_variable"]=="preprocessing_policy"
    assert p8["expected"]["decision"]=="change_tokenizer"
    assert "approval" in p8["situation"].lower()
def test_judge_contains_hard_gate_and_stage_boundary_cases():
    m=load();p=m.build_pack();rows=p["partitions"]["judge"]["held_out"]
    j1=next(x for x in rows if x["root_case_id"]=="J1")
    j8=next(x for x in rows if x["root_case_id"]=="J8")
    assert j1["expected"]["action"]=="reject_checkpoint"
    assert j8["expected"]["action"]=="continue_from_checkpoint"
def test_hash_is_deterministic():
    m=load();a=m.build_pack();b=m.build_pack()
    assert m.canonical_sha256(a)==m.canonical_sha256(b)

def test_launch_authorization_is_consistent():
    import json
    cfg=json.loads((ROOT/"configs/experiments/hephaestus_planner_judge_bakeoff_v1.json").read_text())
    marker=json.loads((ROOT/"configs/experiments/planner_judge_bakeoff_v1.launch.json").read_text())
    assert bool(cfg["governance"]["paid_launch_allowed"]) == bool(marker["authorized"])
    if marker["authorized"]:
        assert marker["authorization_source"] == "user_directive"
        assert marker["authorization_text"].strip()

def test_no_automatic_selection_or_promotion():
    import json
    cfg=json.loads((ROOT/"configs/experiments/hephaestus_planner_judge_bakeoff_v1.json").read_text())
    assert cfg["comparison"]["no_automatic_selection_commit"] is True
    assert cfg["comparison"]["no_automatic_promotion"] is True

def test_pinned_pack_hash():
    import json
    m=load();p=m.build_pack()
    cfg=json.loads((ROOT/"configs/experiments/hephaestus_planner_judge_bakeoff_v1.json").read_text())
    assert m.canonical_sha256(p)==cfg["pack"]["canonical_sha256"]=="1723aa1da329e0440d27073a62f74c1e0666a7649c55a00f9f9f55d6400ed749"
