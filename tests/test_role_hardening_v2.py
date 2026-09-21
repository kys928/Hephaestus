import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("hardening", ROOT / "scripts/build_role_hardening_v2.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hardening_pack_validates_and_keeps_frozen_redteam_final_only():
    mod = _load()
    pack = mod.build_pack()
    mod.validate(pack)
    assert pack["frozen_redteam_optimization_use"] == "forbidden"
    assert len(mod.canonical_sha256(pack)) == 64


def test_training_is_targeted_plus_rehearsal_and_cert_is_targeted_only():
    mod = _load()
    pack = mod.build_pack()
    for role in mod.TARGET_COUNTS:
        train = pack["splits"][role]["train"]
        dev = pack["splits"][role]["dev"]
        cert = pack["splits"][role]["cert"]
        assert any(row["hardening_kind"] == "targeted_neighbor" for row in train)
        assert any(row["hardening_kind"] == "role_mastery_rehearsal" for row in train)
        assert any(row["hardening_kind"] == "role_mastery_rehearsal" for row in dev)
        assert {row["hardening_kind"] for row in cert} == {"targeted_neighbor"}


def test_targeted_cases_externalize_machine_facts():
    mod = _load()
    pack = mod.build_pack()
    for role, splits in pack["splits"].items():
        targeted = [x for x in splits["train"] if x["hardening_kind"] == "targeted_neighbor"]
        assert targeted
        assert all(isinstance(x["verified_machine_facts"], dict) for x in targeted)
        assert any(x["verified_machine_facts"] for x in targeted), role
