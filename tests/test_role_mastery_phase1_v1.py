import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load():
 p=ROOT/"scripts/build_role_mastery_phase1_v1.py";s=importlib.util.spec_from_file_location("rm",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_pack_valid_and_counts():
 m=load();p=m.build_pack();m.validate(p)
 for role in m.SKILLS:
  assert len(p["partitions"][role]["train"])==600
  assert len(p["partitions"][role]["dev"])==100
  assert len(p["partitions"][role]["cert"])==150
def test_all_roles_have_ten_skills():
 m=load();assert all(len(v)==10 for v in m.SKILLS.values())
def test_split_ids_and_evidence_refs_are_disjoint():
 m=load();p=m.build_pack()
 for role in m.SKILLS:
  ids={s:{x["case_id"] for x in p["partitions"][role][s]} for s in m.SPLIT_COUNTS}
  assert ids["train"].isdisjoint(ids["dev"]) and ids["train"].isdisjoint(ids["cert"]) and ids["dev"].isdisjoint(ids["cert"])
  refs={s:{e["ref"] for x in p["partitions"][role][s] for e in x["evidence"]} for s in m.SPLIT_COUNTS}
  assert refs["train"].isdisjoint(refs["cert"]) and refs["dev"].isdisjoint(refs["cert"])
def test_cert_surfaces_are_independent():
 m=load();p=m.build_pack()
 for role in m.SKILLS:
  assert all("Certification" in x["situation"] or "certification" in x["situation"].lower() or "previously unseen" in x["situation"].lower() for x in p["partitions"][role]["cert"])
def test_hash_deterministic():
 m=load();assert m.canonical_sha256(m.build_pack())==m.canonical_sha256(m.build_pack())
