"""Deterministic, policy-conservative model candidate selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.discovery_contract import (
    ModelCandidate,
    ModelSearchRequest,
    ModelSelectionDecision,
)


def _string_set(value: object) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item) for item in value}


def _number(mapping: Mapping[str, object], key: str) -> float | None:
    value = mapping.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class DeterministicModelSelectionService:
    """Rank compatible candidates while preserving every rejection reason."""

    require_revision: bool = True
    require_known_license: bool = True

    def select(
        self, request: ModelSearchRequest, candidates: Sequence[ModelCandidate]
    ) -> ModelSelectionDecision:
        ordered = sorted(candidates, key=lambda candidate: candidate.candidate_id)
        rejected: dict[str, str] = {}
        scores: dict[str, dict[str, float]] = {}
        eligible: list[ModelCandidate] = []
        issues: list[ContractIssue] = []

        for candidate in ordered:
            reasons = self._rejection_reasons(request, candidate)
            if reasons:
                rejected[candidate.candidate_id] = ";".join(reasons)
                continue
            scores[candidate.candidate_id] = self._score(request, candidate)
            eligible.append(candidate)

        ranked = sorted(
            eligible,
            key=lambda item: (-scores[item.candidate_id]["total"], item.candidate_id),
        )
        if not candidates:
            issues.append(ContractIssue(
                code="model_candidates_missing",
                category="candidate_not_found",
                message="No model provider returned a candidate.",
                retryable=True,
                blocking=True,
                evidence_refs=list(request.evidence_refs),
            ))
        elif not ranked:
            categories = {reason for value in rejected.values() for reason in value.split(";")}
            category = "license_unknown" if "license_unknown" in categories else "incompatible_candidate"
            issues.append(ContractIssue(
                code="no_eligible_model",
                category=category,
                message="All discovered model candidates were rejected by explicit constraints.",
                blocking=True,
                evidence_refs=list(request.evidence_refs),
                metadata={"rejected_candidates": dict(rejected)},
            ))

        selected = ranked[0] if ranked else None
        approvals = sorted({
            str(item)
            for item in (selected.metadata.get("required_approvals", []) if selected else [])
        })
        return ModelSelectionDecision(
            decision_id=f"model-selection-{request.request_id}",
            request_id=request.request_id,
            status="selected" if selected else "blocked",
            selected_candidate_id=selected.candidate_id if selected else None,
            ranked_candidate_ids=[item.candidate_id for item in ranked],
            rejected_candidates=rejected,
            selection_rationale=(
                "Selected the highest deterministic score among candidates that passed license, revision, "
                "architecture, tokenizer, context, compute, backend, provenance, integrity, and smoke-test checks."
                if selected else "No candidate passed all mandatory compatibility and governance checks."
            ),
            required_approvals=approvals,
            evidence_refs=sorted({ref for item in ranked for ref in item.evidence_refs} | set(request.evidence_refs)),
            issues=issues,
            confidence=round(min(0.95, scores[selected.candidate_id]["total"]), 6) if selected else 0.0,
            metadata={"score_components": scores, "candidate_count": len(candidates)},
        )

    def _rejection_reasons(self, request: ModelSearchRequest, candidate: ModelCandidate) -> list[str]:
        reasons: list[str] = []
        missing_capabilities = set(request.task_requirements) - set(candidate.capabilities)
        if missing_capabilities:
            reasons.append("task_capability_missing")
        if self.require_revision and not candidate.revision:
            reasons.append("revision_unavailable")
        if candidate.compatibility.get("requested_revision") is False:
            reasons.append("revision_unavailable")
        if self.require_known_license and not candidate.license:
            reasons.append("license_unknown")
        if request.license_allowlist and candidate.license not in request.license_allowlist:
            reasons.append("license_not_allowed")

        architecture = request.architecture_constraints
        allowed_families = _string_set(architecture.get("allowed_families"))
        required_family = str(architecture.get("family", "")).strip()
        if required_family and candidate.architecture_family != required_family:
            reasons.append("architecture_incompatible")
        if allowed_families and candidate.architecture_family not in allowed_families:
            reasons.append("architecture_incompatible")
        max_parameters = _number(architecture, "max_parameters") or _number(request.budget_constraints, "max_parameters")
        if max_parameters is not None and (candidate.parameter_count is None or candidate.parameter_count > max_parameters):
            reasons.append("parameter_budget_exceeded")

        tokenizer = request.tokenizer_constraints
        required_tokenizer = str(tokenizer.get("tokenizer_ref", "")).strip()
        allowed_tokenizers = _string_set(tokenizer.get("allowed_tokenizers"))
        if required_tokenizer and candidate.tokenizer_ref != required_tokenizer:
            reasons.append("tokenizer_incompatible")
        if allowed_tokenizers and candidate.tokenizer_ref not in allowed_tokenizers:
            reasons.append("tokenizer_incompatible")

        minimum_context = _number(architecture, "min_context_length") or _number(request.runtime_constraints, "min_context_length")
        if minimum_context is not None and (candidate.context_length is None or candidate.context_length < minimum_context):
            reasons.append("context_too_short")

        requested_backend = str(request.runtime_constraints.get("backend", "")).strip()
        supported = _string_set(candidate.runtime_requirements.get("supported_backends"))
        if requested_backend and (not supported or requested_backend not in supported):
            reasons.append("backend_incompatible")
        max_memory = _number(request.runtime_constraints, "max_memory_gb")
        memory = _number(candidate.runtime_requirements, "memory_gb")
        if max_memory is not None and (memory is None or memory > max_memory):
            reasons.append("memory_budget_exceeded")
        max_runtime = _number(request.budget_constraints, "max_runtime_seconds")
        runtime = _number(candidate.runtime_requirements, "estimated_runtime_seconds")
        if max_runtime is not None and (runtime is None or runtime > max_runtime):
            reasons.append("runtime_budget_exceeded")

        if candidate.compatibility.get("runtime") is False or candidate.compatibility.get("compatible") is False:
            reasons.append("runtime_incompatible")
        required_integrity = str(request.runtime_constraints.get("checkpoint_integrity", "")).strip()
        if required_integrity and candidate.compatibility.get("checkpoint_integrity") != required_integrity:
            reasons.append("checkpoint_integrity_insufficient")
        if request.runtime_constraints.get("smoke_test_required") is True and candidate.compatibility.get("smoke_test") is not True:
            reasons.append("smoke_test_unsuitable")
        if (
            "provenance_unknown" in candidate.risk_signals
            or "provenance" in candidate.missing_metadata
            or not candidate.artifact_ref
            or not candidate.evidence_refs
        ):
            reasons.append("provenance_unknown")
        return sorted(set(reasons))

    def _score(self, request: ModelSearchRequest, candidate: ModelCandidate) -> dict[str, float]:
        required_tasks = set(request.task_requirements)
        capabilities = set(candidate.capabilities)
        task_match = len(required_tasks & capabilities) / len(required_tasks) if required_tasks else 1.0
        min_context = _number(request.runtime_constraints, "min_context_length") or 1.0
        context_fit = min(1.0, float(candidate.context_length or 0) / min_context)
        runtime_fit = 1.0 if candidate.runtime_requirements.get("supported_backends") else 0.5
        evidence = 1.0 if candidate.evidence_refs and candidate.artifact_ref else 0.5
        risk = max(0.0, 1.0 - 0.2 * len(candidate.risk_signals))
        total = 0.35 * task_match + 0.20 * context_fit + 0.20 * runtime_fit + 0.15 * evidence + 0.10 * risk
        return {
            "task_match": round(task_match, 6),
            "context_fit": round(context_fit, 6),
            "runtime_fit": round(runtime_fit, 6),
            "evidence": round(evidence, 6),
            "risk": round(risk, 6),
            "total": round(total, 6),
        }
