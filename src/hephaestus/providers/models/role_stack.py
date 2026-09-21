"""Certified role-model stack registry.

This module exposes the independently certified Hephaestus role adapters as a
strict, read-only runtime registry. It deliberately does not replace control-
spine role implementations or perform network/model loading by itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

REQUIRED_ROLES = ("controller", "diagnosis", "planner", "evaluator", "judge")
_ALLOWED_LICENSES = {"apache-2.0", "mit"}
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class AdapterArtifact:
    s3_key: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class RoleCertification:
    passed: bool
    quality_100: float
    schema_compliance: float
    evidence_grounding: float
    hallucination_rate: float
    observed_min_skill_quality: float


@dataclass(frozen=True, slots=True)
class CertifiedRoleModel:
    role: str
    model_id: str
    revision: str
    license: str
    status: str
    selected_dev_step: int
    adapter: AdapterArtifact
    certification: RoleCertification


@dataclass(frozen=True, slots=True)
class CertifiedRoleModelStack:
    stack_id: str
    status: str
    production_certified: bool
    final_role_adapters_trained: bool
    runtime_registry_ready: bool
    automatic_role_dispatch_enabled: bool
    source_run_id: str
    frozen_repo_sha: str
    evidence_record: str
    roles: Mapping[str, CertifiedRoleModel]

    def resolve(self, role: str) -> CertifiedRoleModel:
        try:
            return self.roles[role]
        except KeyError as exc:
            raise KeyError(f"unknown certified Hephaestus role: {role}") from exc


def _number(value: object, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


def _positive_int(value: object, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _validate_role(role: str, payload: object) -> CertifiedRoleModel:
    if not isinstance(payload, dict):
        raise ValueError(f"role {role} must be an object")

    model_id = str(payload.get("model_id", "")).strip()
    revision = str(payload.get("revision", "")).strip()
    license_id = str(payload.get("license", "")).strip().lower()
    status = str(payload.get("status", "")).strip()
    if not model_id:
        raise ValueError(f"role {role} is missing model_id")
    if not _REVISION_RE.fullmatch(revision):
        raise ValueError(f"role {role} revision must be an immutable 40-character commit SHA")
    if license_id not in _ALLOWED_LICENSES:
        raise ValueError(f"role {role} license is not in the certified allowlist")
    if status != "certified":
        raise ValueError(f"role {role} is not certified")

    adapter_raw = payload.get("adapter")
    if not isinstance(adapter_raw, dict):
        raise ValueError(f"role {role} adapter must be an object")
    s3_key = str(adapter_raw.get("s3_key", "")).strip()
    sha256 = str(adapter_raw.get("sha256", "")).strip()
    adapter_bytes = _positive_int(adapter_raw.get("bytes"), f"{role}.adapter.bytes")
    if not s3_key.startswith("hephaestus/scientific/v1/role_mastery/"):
        raise ValueError(f"role {role} adapter must reference governed role-mastery storage")
    if not _SHA256_RE.fullmatch(sha256):
        raise ValueError(f"role {role} adapter SHA-256 is malformed")

    cert_raw = payload.get("certification")
    if not isinstance(cert_raw, dict):
        raise ValueError(f"role {role} certification must be an object")
    passed = cert_raw.get("passed") is True
    if not passed:
        raise ValueError(f"role {role} certification did not pass")
    certification = RoleCertification(
        passed=True,
        quality_100=_number(cert_raw.get("quality_100"), f"{role}.quality_100"),
        schema_compliance=_number(cert_raw.get("schema_compliance"), f"{role}.schema_compliance"),
        evidence_grounding=_number(cert_raw.get("evidence_grounding"), f"{role}.evidence_grounding"),
        hallucination_rate=_number(cert_raw.get("hallucination_rate"), f"{role}.hallucination_rate"),
        observed_min_skill_quality=_number(
            cert_raw.get("observed_min_skill_quality"),
            f"{role}.observed_min_skill_quality",
        ),
    )
    if certification.schema_compliance != 1.0:
        raise ValueError(f"role {role} certification lacks perfect schema compliance")
    if not 0.0 <= certification.evidence_grounding <= 1.0:
        raise ValueError(f"role {role} evidence_grounding is outside [0,1]")
    if not 0.0 <= certification.hallucination_rate <= 1.0:
        raise ValueError(f"role {role} hallucination_rate is outside [0,1]")

    return CertifiedRoleModel(
        role=role,
        model_id=model_id,
        revision=revision,
        license=license_id,
        status=status,
        selected_dev_step=_positive_int(payload.get("selected_dev_step"), f"{role}.selected_dev_step"),
        adapter=AdapterArtifact(s3_key=s3_key, sha256=sha256, bytes=adapter_bytes),
        certification=certification,
    )


def load_certified_role_model_stack(path: str | Path) -> CertifiedRoleModelStack:
    """Load and strictly validate the certified role-model registry."""

    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("role-model stack must be a JSON object")
    if payload.get("status") != "certified_role_adapters_v1":
        raise ValueError("role-model stack is not in certified_role_adapters_v1 state")
    for flag in ("production_certified", "final_role_adapters_trained", "runtime_registry_ready"):
        if payload.get(flag) is not True:
            raise ValueError(f"role-model stack requires {flag}=true")

    raw_roles = payload.get("roles")
    if not isinstance(raw_roles, dict):
        raise ValueError("role-model stack roles must be an object")
    if set(raw_roles) != set(REQUIRED_ROLES):
        raise ValueError("role-model stack must contain exactly the five certified Hephaestus roles")
    roles = {role: _validate_role(role, raw_roles[role]) for role in REQUIRED_ROLES}

    source = payload.get("source_experiment")
    if not isinstance(source, dict):
        raise ValueError("role-model stack source_experiment must be an object")
    source_run_id = str(source.get("run_id", "")).strip()
    frozen_repo_sha = str(source.get("frozen_repo_sha", "")).strip()
    if not source_run_id:
        raise ValueError("role-model stack source run id is missing")
    if not _REVISION_RE.fullmatch(frozen_repo_sha):
        raise ValueError("role-model stack frozen repo SHA is malformed")

    evidence_record = str(payload.get("evidence_record", "")).strip()
    if not evidence_record:
        raise ValueError("role-model stack evidence_record is missing")

    return CertifiedRoleModelStack(
        stack_id=str(payload.get("stack_id", "")).strip(),
        status=str(payload["status"]),
        production_certified=True,
        final_role_adapters_trained=True,
        runtime_registry_ready=True,
        automatic_role_dispatch_enabled=payload.get("automatic_role_dispatch_enabled") is True,
        source_run_id=source_run_id,
        frozen_repo_sha=frozen_repo_sha,
        evidence_record=evidence_record,
        roles=roles,
    )
