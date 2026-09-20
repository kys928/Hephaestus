from pathlib import Path
import importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
def load_builder():
    p=ROOT/"scripts/build_diagnosis_foundation_bakeoff_v1.py";s=importlib.util.spec_from_file_location("dfb",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_pack_is_disjoint_and_balanced():
    m=load_builder();p=m.build_pack();m.validate(p)
    assert len(p["partitions"]["zero_shot"])==30
    assert len(p["partitions"]["micro_lora_train"])==72
    assert len(p["partitions"]["held_out"])==48
    for split in p["partitions"]:
        roots={r:sum(x["root_case_id"]==r for x in p["partitions"][split]) for r in ["D1","D2","D3","D4","D5","D6"]}
        assert len(set(roots.values()))==1
def test_positive_and_uncertain_causality_both_exist():
    p=load_builder().build_pack();rows=p["partitions"]["zero_shot"]
    d1=[x for x in rows if x["root_case_id"]=="D1"]
    assert any(x["expected"]["decision"]=="inconclusive" for x in d1)
    assert any(x["expected"]["decision"]=="data_quality" for x in d1)
def test_only_clean_foundations_are_stage2_candidates():
    c=json.loads((ROOT/"configs/experiments/hephaestus_diagnosis_foundation_bakeoff_v1.json").read_text())
    assert [x["candidate_id"] for x in c["candidates"] if x["adaptation_candidate"]]==["granite42-3b","ministral3-3b-reasoning"]
    assert c["stage2_plan"]["paid_launch_allowed_now"] is False
