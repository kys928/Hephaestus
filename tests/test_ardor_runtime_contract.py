from hephaestus.backends.ardor.runtime_contract import normalize_ardor_runtime_contract


def test_normalize_v1_contract() -> None:
    payload={"contract_version":"ardor_runtime_contract.v1","run_id":"r1","status":"succeeded","artifacts":{"metrics_ref":"m","deterministic_ref":"d"},"checkpoint_candidates":[{"checkpoint_ref":"c1","content_hash":"h","hash_type":"sha256"},{"checkpoint_ref":"c2"}]}
    out=normalize_ardor_runtime_contract(payload)
    assert out["contract_integrity_level"] in {"complete","partial"}
    assert out["checkpoint_candidates"][0]["integrity_level"]=="content_hash"
    assert out["checkpoint_candidates"][1]["integrity_level"]=="ref"


def test_normalize_legacy_contract() -> None:
    payload={"run_id":"r1","status":"succeeded","artifacts":{"metrics_ref":"m","deterministic_ref":"d","checkpoint_refs":["c1"]},"checkpoint_scores":{"c1":0.7}}
    out=normalize_ardor_runtime_contract(payload)
    assert out["contract_integrity_level"] in {"legacy","partial","complete"}
    assert "legacy_contract_shape" in out["warnings"]
    assert out["checkpoint_candidates"][0]["checkpoint_ref"]=="c1"
