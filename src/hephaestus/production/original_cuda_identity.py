"""Immutable identity checks for the five original certified CUDA roles.

No model execution, source approval, or publication occurs here.
"""
from hashlib import sha256
import json
from pathlib import Path

from hephaestus.llm.role_contract import RoleMasteryContract
from hephaestus.providers.models.role_stack import REQUIRED_ROLES, load_certified_role_model_stack

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs/models/hephaestus_role_model_stack_v1.json"
CONTRACT = ROOT / "configs/models/hephaestus_role_mastery_runtime_v1.json"
CERTIFICATE = ROOT / "docs/evidence/role-model-stack-v1/certification.json"


def registry_digest():
    return "sha256:" + sha256(REGISTRY.read_bytes()).hexdigest()


def verified_original_roles(expected_registry_digest):
    if expected_registry_digest != registry_digest():
        raise PermissionError("certified CUDA role registry differs from trusted digest")
    original = json.loads(CERTIFICATE.read_bytes())
    stack = load_certified_role_model_stack(REGISTRY)
    contract = RoleMasteryContract.load(CONTRACT)
    if (original.get("status") != "certified_and_artifact_verified"
            or set(original.get("roles", {})) != set(REQUIRED_ROLES)
            or set(stack.roles) != set(REQUIRED_ROLES)
            or stack.automatic_role_dispatch_enabled is not False):
        raise PermissionError("original frozen CUDA role identity boundary changed")
    for role in REQUIRED_ROLES:
        model = stack.resolve(role)
        contract.validate_model(model)
        source = original["roles"][role]
        if ((model.model_id, model.revision, model.adapter.sha256) !=
            (source.get("model_id"), source.get("revision"), source.get("adapter_sha256"))):
            raise PermissionError("original CUDA role model or adapter changed")
    if len({(m.model_id,m.revision,m.adapter.sha256) for m in stack.roles.values()}) != 5:
        raise PermissionError("five distinct original CUDA model identities required")
    return stack, contract
