from __future__ import annotations

import json
from pathlib import Path

import pytest

from hephaestus.providers.models.role_stack import (
    REQUIRED_ROLES,
    load_certified_role_model_stack,
)


ROOT = Path(__file__).resolve().parents[1]
STACK = ROOT / "configs/models/hephaestus_role_model_stack_v1.json"


def test_certified_role_stack_loads_all_five_roles() -> None:
    stack = load_certified_role_model_stack(STACK)
    assert stack.production_certified is True
    assert stack.final_role_adapters_trained is True
    assert stack.runtime_registry_ready is True
    assert stack.automatic_role_dispatch_enabled is False
    assert tuple(stack.roles) == REQUIRED_ROLES
    for role in REQUIRED_ROLES:
        spec = stack.resolve(role)
        assert spec.status == "certified"
        assert spec.certification.passed is True
        assert spec.certification.schema_compliance == 1.0
        assert spec.adapter.bytes > 0
        assert len(spec.adapter.sha256) == 64
        assert len(spec.revision) == 40


def test_certified_role_stack_rejects_tampered_adapter_digest(tmp_path: Path) -> None:
    payload = json.loads(STACK.read_text(encoding="utf-8"))
    payload["roles"]["judge"]["adapter"]["sha256"] = "not-a-digest"
    path = tmp_path / "stack.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_certified_role_model_stack(path)


def test_certified_role_stack_rejects_uncertified_role(tmp_path: Path) -> None:
    payload = json.loads(STACK.read_text(encoding="utf-8"))
    payload["roles"]["planner"]["certification"]["passed"] = False
    path = tmp_path / "stack.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="certification did not pass"):
        load_certified_role_model_stack(path)
