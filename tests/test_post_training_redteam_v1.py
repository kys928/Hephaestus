import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / "scripts/build_post_training_redteam_v1.py"
    spec = importlib.util.spec_from_file_location("post_training_redteam", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_post_training_redteam_pack_is_valid_and_frozen_shape():
    mod = load_builder()
    pack = mod.build_pack()
    mod.validate(pack)
    assert sum(len(rows) for rows in pack["cases"].values()) == 600
    assert set(pack["cases"]) == set(mod.ROLES)
    assert len(mod.canonical_sha256(pack)) == 64


def test_every_role_has_new_ood_dimensions_and_stressors():
    mod = load_builder()
    pack = mod.build_pack()
    for role, rows in pack["cases"].items():
        dimensions = {row["dimension"] for row in rows}
        assert len(dimensions) >= 20  # 12 OOD families plus counterfactual sides
        stressors = {s for row in rows for s in row["stressors"]}
        assert "prompt_injection_in_evidence" in stressors
        assert "cross_role_bait" in stressors
        assert "long_context_noise" in stressors
        assert "one_fact_counterfactual" in stressors


def test_invariance_pairs_preserve_targets_and_flip_pairs_change_targets():
    mod = load_builder()
    pack = mod.build_pack()
    for rows in pack["cases"].values():
        pairs = {}
        for row in rows:
            pairs.setdefault(row["pair_id"], []).append(row)
        for pair in pairs.values():
            assert len(pair) == 2
            left, right = pair
            l = left["expected"]
            r = right["expected"]
            lt = (l["decision"], l["action"], l["primary_variable"])
            rt = (r["decision"], r["action"], r["primary_variable"])
            if left["pair_relation"] == "same_semantics":
                assert lt == rt
            else:
                assert left["pair_relation"] == "one_fact_flip"
                assert lt != rt
