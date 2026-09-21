import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def mod():
 p=ROOT/"scripts/build_role_mastery_v1.py";s=importlib.util.spec_from_file_location("rm",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_role_mastery_pack_valid():
 m=mod();p=m.build_pack();m.validate(p)
 assert sum(len(v) for r in p["splits"].values() for v in r.values())==9800
def test_cert_is_disjoint():
 m=mod();p=m.build_pack()
 for role in p["splits"]:
  a={x["case_id"] for x in p["splits"][role]["train"]};b={x["case_id"] for x in p["splits"][role]["dev"]};c={x["case_id"] for x in p["splits"][role]["cert"]}
  assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)
def test_all_skills_each_split():
 m=mod();p=m.build_pack()
 for role in p["splits"]:
  skills={x["skill"] for x in m.FAMILIES[role]}
  for split in ("train","dev","cert"): assert {x["skill"] for x in p["splits"][role][split]}==skills
def test_targets_use_only_admissible_evidence():
 m=mod();p=m.build_pack()
 for role in p["splits"]:
  for split in p["splits"][role]:
   rows=p["splits"][role][split]
   for row in rows[::max(1,len(rows)//20)]:
    assert set(m.target(row)["evidence_refs"])==set(row["allowed_evidence_refs"])
