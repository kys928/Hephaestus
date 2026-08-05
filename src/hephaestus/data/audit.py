from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSearchRequest


def _tokens(values: list[str]) -> set[str]:
    return {
        token
        for value in values
        for token in re.findall(r"[a-z0-9][a-z0-9_+.-]*", str(value).casefold())
        if len(token) > 1
    }


def _fit(required: list[str], available: list[str]) -> float:
    wanted = _tokens(required)
    if not wanted:
        return 1.0
    offered = _tokens(available)
    if not offered:
        return 0.25
    return len(wanted & offered) / len(wanted)


@dataclass(slots=True)
class CandidateAudit:
    candidate_id: str
    audit_scope: str = "metadata_only"
    score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)
    rejected_reasons: list[str] = field(default_factory=list)
    required_approvals: list[str] = field(default_factory=list)
    preprocessing_requirements: list[str] = field(default_factory=list)
    missing_metadata: list[str] = field(default_factory=list)
    contamination_status: str = "not_checked"
    synthetic_status: str = "not_synthetic"
    support_set_status: str = "not_support_set"
    hard_negative_status: str = "not_hard_negative"
    estimated_cost: dict[str, int | None] = field(default_factory=dict)
    uncertainty: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_candidate(request: DatasetSearchRequest, candidate: DatasetCandidate) -> CandidateAudit:
    searchable = [
        candidate.dataset_id,
        *candidate.task_types,
        *candidate.languages,
        *candidate.domains,
        str(candidate.metadata.get("description", "")),
        str(candidate.metadata.get("capability_targets", "")),
    ]
    relevance_targets = [request.problem_statement, *request.capability_targets]
    relevance_tokens = _tokens(relevance_targets)
    searchable_tokens = _tokens(searchable)
    relevance = len(relevance_tokens & searchable_tokens) / max(len(relevance_tokens), 1)
    task_fit = _fit(request.capability_targets, candidate.task_types + candidate.domains)
    language_fit = _fit(request.required_languages, candidate.languages)
    domain_fit = _fit(request.required_domains, candidate.domains)
    record_format = str(candidate.format_profile.get("record_format", ""))
    format_fit = _fit(request.required_formats, [record_format, *map(str, candidate.format_profile.values())])

    compatibility_flags = {
        key: value
        for key, value in candidate.compatibility.items()
        if key in {"compatible", "model_compatible", "tokenizer_compatible", "local_readable"}
    }
    false_compatibility = sorted(key for key, value in compatibility_flags.items() if value is False)
    remote_code_required = bool(candidate.compatibility.get("remote_code_required", False))
    compatibility = 0.0 if false_compatibility or remote_code_required else 1.0

    allowlist = {item.strip().casefold() for item in request.license_allowlist if item.strip()}
    denylist = {item.strip().casefold() for item in request.license_denylist if item.strip()}
    license_name = candidate.license.casefold() if candidate.license else None
    license_score = 1.0 if license_name else 0.0
    provenance_score = 1.0 if candidate.provenance else 0.0
    trust_scores = {"verified": 1.0, "internal": 1.0, "local_fixture": 0.9, "external_metadata": 0.65, "unknown": 0.25, "untrusted": 0.0}
    trust_score = trust_scores.get(candidate.trust_level, 0.4)

    max_rows = request.size_constraints.get("max_rows")
    min_rows = request.size_constraints.get("min_rows")
    max_bytes = request.size_constraints.get("max_bytes")
    cost_score = 1.0
    if candidate.estimated_bytes is not None and max_bytes:
        cost_score = min(1.0, float(max_bytes) / max(candidate.estimated_bytes, 1))
    elif candidate.estimated_rows is not None and max_rows:
        cost_score = min(1.0, float(max_rows) / max(candidate.estimated_rows, 1))

    missing = set(candidate.missing_metadata)
    if candidate.estimated_rows is None:
        missing.add("estimated_rows")
    if candidate.estimated_bytes is None:
        missing.add("estimated_bytes")
    uncertainty = min(1.0, 0.12 * len(missing) + 0.08 * len(candidate.risk_signals))
    certainty = 1.0 - uncertainty
    expected_benefit = min(1.0, (relevance + task_fit + language_fit + domain_fit + format_fit) / 5.0)

    components = {
        "relevance": relevance,
        "task_fit": task_fit,
        "language_fit": language_fit,
        "domain_fit": domain_fit,
        "format_fit": format_fit,
        "compatibility": compatibility,
        "license": license_score,
        "provenance": provenance_score,
        "trust": trust_score,
        "cost_efficiency": cost_score,
        "expected_benefit": expected_benefit,
        "certainty": certainty,
    }
    weights = {
        "relevance": 0.16,
        "task_fit": 0.12,
        "language_fit": 0.08,
        "domain_fit": 0.08,
        "format_fit": 0.08,
        "compatibility": 0.14,
        "license": 0.08,
        "provenance": 0.07,
        "trust": 0.06,
        "cost_efficiency": 0.04,
        "expected_benefit": 0.06,
        "certainty": 0.03,
    }
    risk_penalty = min(0.35, 0.04 * len(candidate.risk_signals))
    score = max(0.0, sum(components[key] * weights[key] for key in weights) - risk_penalty)

    rejected: list[str] = []
    approvals: list[str] = []
    preprocessing: list[str] = []
    if not candidate.dataset_id:
        rejected.append("dataset_id_missing")
    if false_compatibility:
        rejected.extend(f"incompatible:{key}" for key in false_compatibility)
    if remote_code_required:
        rejected.append("remote_dataset_code_not_allowed")
    if license_name in denylist:
        rejected.append(f"license_denied:{license_name}")
    elif allowlist and license_name and license_name not in allowlist:
        rejected.append(f"license_not_allowlisted:{license_name}")
    elif not license_name:
        approvals.append(f"unknown_license:{candidate.candidate_id}")
    if not candidate.provenance:
        approvals.append(f"unknown_provenance:{candidate.candidate_id}")
    if candidate.trust_level in {"unknown", "untrusted"}:
        approvals.append(f"low_trust:{candidate.candidate_id}")
    if max_rows is not None and candidate.estimated_rows is not None and candidate.estimated_rows > int(max_rows):
        rejected.append("estimated_rows_exceed_limit")
    if min_rows is not None and candidate.estimated_rows is not None and candidate.estimated_rows < int(min_rows):
        rejected.append("estimated_rows_below_minimum")
    if max_bytes is not None and candidate.estimated_bytes is not None and candidate.estimated_bytes > int(max_bytes):
        rejected.append("estimated_bytes_exceed_limit")

    if record_format not in {"jsonl", "json", "csv"}:
        preprocessing.append("format_adapter_required")
    if candidate.metadata.get("requires_normalization", True):
        preprocessing.append("unicode_and_whitespace_normalization")
    preprocessing.extend(str(item) for item in candidate.metadata.get("preprocessing", []) if str(item))
    if request.tokenizer_ref:
        preprocessing.append("tokenizer_compatibility_validation")

    contamination_status = str(candidate.metadata.get("contamination_status", "not_checked"))
    if contamination_status not in {"not_checked", "partially_checked", "checked"}:
        contamination_status = "not_checked"
    return CandidateAudit(
        candidate_id=candidate.candidate_id,
        score=round(score, 8) if math.isfinite(score) else 0.0,
        score_components={key: round(value, 8) for key, value in components.items()},
        rejected_reasons=sorted(set(rejected)),
        required_approvals=sorted(set(approvals)),
        preprocessing_requirements=sorted(set(preprocessing)),
        missing_metadata=sorted(missing),
        contamination_status=contamination_status,
        synthetic_status="synthetic" if candidate.metadata.get("synthetic") else "not_synthetic",
        support_set_status="support_set" if candidate.metadata.get("support_set") else "not_support_set",
        hard_negative_status="hard_negative" if candidate.metadata.get("hard_negative") else "not_hard_negative",
        estimated_cost={"rows": candidate.estimated_rows, "bytes": candidate.estimated_bytes},
        uncertainty=round(uncertainty, 8),
    )
