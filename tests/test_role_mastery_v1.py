import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def mod():
 p=ROOT/"scripts/build_role_mastery_v1.py";s=importlib.util.spec_from_file_location("rm",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_pack():
 m=mod();p=m.build_pack();m.validate(p);assert len(p["partitions"])==5
 assert all(len(p["partitions"][r]["train"])==1024 for r in m.ROLES)
 assert all(len(p["partitions"][r]["certification"])==128 for r in m.ROLES)
def test_disjoint():
 m=mod();p=m.build_pack()
 for r in m.ROLES:
  assert {x["case_id"] for x in p["partitions"][r]["train"]}.isdisjoint({x["case_id"] for x in p["partitions"][r]["certification"]})
def test_hash():
 m=mod();assert m.canonical_sha256(m.build_pack())==m.canonical_sha256(m.build_pack())
