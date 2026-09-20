from pathlib import Path
import importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
def load_runner():
    p=ROOT/"scripts/run_diagnosis_adaptability_v1.py";s=importlib.util.spec_from_file_location("da",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def load_pack():
    p=ROOT/"scripts/build_diagnosis_foundation_bakeoff_v1.py";s=importlib.util.spec_from_file_location("pb",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);pack=m.build_pack();m.validate(pack);return pack
def cfg(): return json.loads((ROOT/"configs/experiments/hephaestus_diagnosis_adaptability_v1.json").read_text())
def test_matched_budget_and_hyperparameters():
    c=cfg();t=c["training"]
    assert t["lora_rank"]==8 and t["lora_alpha"]==16 and t["optimizer_steps"]==32
    assert t["learning_rate"]==5e-5 and t["seed"]==11
    assert t["padded_training_token_slots"]==t["optimizer_steps"]*t["gradient_accumulation_steps"]*t["max_sequence_length"]==98304
def test_training_and_held_out_are_disjoint():
    p=load_pack();tr={x["case_id"] for x in p["partitions"]["micro_lora_train"]};ho={x["case_id"] for x in p["partitions"]["held_out"]}
    assert len(tr)==72 and len(ho)==48 and tr.isdisjoint(ho)
def test_training_answers_only_use_admissible_evidence():
    m=load_runner();p=load_pack()
    for case in p["partitions"]["micro_lora_train"]:
        ans=json.loads(m.expected_answer(case))
        assert ans["evidence_refs"]==case["allowed_evidence_refs"]
        assert set(ans)==set(m.topo.REQUIRED_KEYS)
def test_d5_training_teaches_material_reference_only():
    m=load_runner();p=load_pack()
    for case in p["partitions"]["micro_lora_train"]:
        if case["root_case_id"]=="D5":
            ans=json.loads(m.expected_answer(case))
            assert len(ans["evidence_refs"])==1
            assert ans["evidence_refs"][0].endswith("-MISS")
def test_ministral_target_excludes_vision():
    c=cfg();m=next(x for x in c["candidates"] if x["candidate_id"]=="ministral3-3b-reasoning")
    assert "language_model" in m["lora_target_regex"]
    assert "vision_tower" not in m["lora_target_regex"]
    assert "model.vision_tower." in m["exclude_module_prefixes"]
def test_no_automatic_promotion():
    assert cfg()["comparison"]["no_automatic_promotion"] is True
