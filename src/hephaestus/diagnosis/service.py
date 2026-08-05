"""Deterministic evidence-based diagnosis service."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from hephaestus.diagnosis.evidence import EvidenceAdapter
from hephaestus.diagnosis.normalization import NormalizedEvidence, normalize_evidence
from hephaestus.diagnosis.rules import build_hypotheses, inconclusive_hypothesis
from hephaestus.policy.diagnosis_policy import DiagnosisPolicy
from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.diagnosis_contract import (
    DiagnosisReport,
    DiagnosisRequest,
    EvidenceObservation,
)


@runtime_checkable
class ExplanationAdapter(Protocol):
    """Optional prose-only adapter; deterministic findings remain authoritative."""

    def explain(self, request: DiagnosisRequest, report: DiagnosisReport) -> str: ...


@dataclass(slots=True)
class EvidenceBasedDiagnosisService:
    evidence_adapters: Sequence[EvidenceAdapter] = field(default_factory=tuple)
    policy: DiagnosisPolicy = field(default_factory=DiagnosisPolicy)
    explanation_adapter: ExplanationAdapter | None = None

    def diagnose(self, request: DiagnosisRequest) -> DiagnosisReport:
        if not isinstance(request, DiagnosisRequest):
            return self._invalid_request_report(request)

        issues: list[ContractIssue] = []
        raw_records: list[dict[str, object]] = []
        for index, item in enumerate(deepcopy(request.observed_failures)):
            if not isinstance(item, dict):
                issues.append(
                    ContractIssue(
                        code=f"malformed_evidence_{index}",
                        category="invalid_request",
                        message=f"observed_failures[{index}] is not a structured evidence record",
                        retryable=False,
                        blocking=False,
                    )
                )
                continue
            raw_records.append(item)

        for adapter in self.evidence_adapters:
            try:
                loaded = adapter.load(request)
            except Exception as exc:  # noqa: BLE001 - injected adapter trust boundary
                issues.append(
                    ContractIssue(
                        code="evidence_adapter_failed",
                        category="provider_unavailable",
                        message=f"Evidence adapter {type(adapter).__name__} failed: {type(exc).__name__}",
                        retryable=True,
                        blocking=False,
                    )
                )
                continue
            for item in loaded:
                if isinstance(item, dict):
                    raw_records.append(deepcopy(item))
                else:
                    try:
                        raw_records.append(deepcopy(dict(item)))
                    except (TypeError, ValueError):
                        issues.append(
                            ContractIssue(
                                code="malformed_adapter_evidence",
                                category="invalid_request",
                                message=f"Evidence adapter {type(adapter).__name__} returned a non-mapping record",
                                blocking=False,
                            )
                        )

        evidence = self._normalize(raw_records, issues)
        missing = _missing_evidence(evidence)
        eval_verified = _eval_integrity_verified(evidence)
        if not eval_verified:
            issues.append(
                ContractIssue(
                    code="eval_integrity_unverified",
                    category="missing_evidence",
                    message="Verified eval-pack and deterministic-scorecard integrity are required for strong downstream confidence",
                    retryable=True,
                    blocking=True,
                    evidence_refs=sorted(
                        {
                            item.source_ref
                            for item in evidence
                            if item.evidence_kind
                            in {"eval_report", "scorecard", "deterministic_scorecard"}
                        }
                    ),
                )
            )
        if request.evidence_refs and not self.evidence_adapters:
            issues.append(
                ContractIssue(
                    code="evidence_adapter_missing",
                    category="missing_evidence",
                    message="Evidence references were supplied but no evidence adapter was configured",
                    retryable=True,
                    blocking=False,
                    evidence_refs=list(request.evidence_refs),
                )
            )

        hypotheses = build_hypotheses(evidence, self.policy, eval_verified)
        status, leading_id, report_confidence = self._select_outcome(
            hypotheses, missing
        )
        if not hypotheses:
            hypotheses = [inconclusive_hypothesis(missing)]
            leading_id = hypotheses[0].hypothesis_id

        observations = [
            EvidenceObservation(
                observation_id=item.observation_id,
                evidence_kind=item.evidence_kind,
                source_ref=item.source_ref,
                summary=item.summary,
                severity=item.severity,
                confidence=item.confidence,
                metadata={
                    "statement_type": "observation",
                    "signals": list(item.signals),
                    "normalized": True,
                },
            )
            for item in evidence
        ]
        report_id = _report_id(request, observations)
        report = DiagnosisReport(
            report_id=report_id,
            request_id=request.request_id,
            run_id=request.run_id,
            lineage_id=request.lineage_id,
            stage_name=request.stage_name,
            status=status,
            observations=observations,
            hypotheses=hypotheses,
            leading_hypothesis_id=leading_id,
            missing_evidence=missing,
            issues=_deduplicate_issues(issues),
            confidence=round(report_confidence, 4),
            metadata={
                "deterministic_layer_authoritative": True,
                "confidence_meaning": "support for the ranked hypothesis, not causal certainty",
                "recommendations_executed": False,
                "diagnostic_order_preserved": True,
                "epistemic_distinctions": {
                    "observation": "normalized content explicitly present in a source record",
                    "inference": "deterministic rule match between recorded signals and a finite failure domain",
                    "hypothesis": "ranked possible explanation that does not assert causation",
                    "recommendation": "diagnostic test or intervention kind proposed for later review, not execution",
                    "confidence": "bounded support from source evidence after contradictions and policy caps",
                },
            },
        )
        self._add_optional_explanation(request, report)
        return report

    def _normalize(
        self,
        raw_records: list[dict[str, object]],
        issues: list[ContractIssue],
    ) -> list[NormalizedEvidence]:
        normalized: dict[str, NormalizedEvidence] = {}
        for index, item in enumerate(raw_records):
            try:
                evidence = normalize_evidence(item, index)
            except (TypeError, ValueError, OverflowError) as exc:
                issues.append(
                    ContractIssue(
                        code=f"evidence_normalization_failed_{index}",
                        category="invalid_request",
                        message=f"Evidence normalization failed: {type(exc).__name__}",
                        blocking=False,
                    )
                )
                continue
            normalized[evidence.observation_id] = evidence
        return [normalized[key] for key in sorted(normalized)]

    def _select_outcome(
        self,
        hypotheses: list,
        missing: list[str],
    ) -> tuple[str, str | None, float]:
        if not hypotheses:
            return "inconclusive", None, 0.0
        leading = hypotheses[0]
        confidence = float(leading.confidence)
        if confidence < self.policy.minimum_hypothesis_confidence:
            return "inconclusive", leading.hypothesis_id, confidence
        if len(hypotheses) > 1:
            margin = confidence - float(hypotheses[1].confidence)
            if (
                margin <= self.policy.close_hypothesis_margin
                and confidence < self.policy.close_hypothesis_confidence_ceiling
            ):
                return (
                    "inconclusive",
                    leading.hypothesis_id,
                    max(0.0, confidence - 0.15),
                )
        return "completed", leading.hypothesis_id, confidence

    def _add_optional_explanation(
        self, request: DiagnosisRequest, report: DiagnosisReport
    ) -> None:
        if self.explanation_adapter is None:
            return
        try:
            explanation = self.explanation_adapter.explain(request, report)
        except Exception as exc:  # noqa: BLE001 - optional prose cannot affect correctness
            report.metadata["explanation_adapter_warning"] = type(exc).__name__
            return
        report.metadata["llm_assisted_explanation"] = str(explanation)
        report.metadata["llm_explanation_authoritative"] = False

    def _invalid_request_report(self, request: object) -> DiagnosisReport:
        request_id = str(getattr(request, "request_id", "invalid-request"))
        run_id = str(getattr(request, "run_id", ""))
        lineage_id = str(getattr(request, "lineage_id", ""))
        stage_name = str(getattr(request, "stage_name", ""))
        return DiagnosisReport(
            report_id=f"diag-invalid-{hashlib.sha256(request_id.encode('utf-8')).hexdigest()[:12]}",
            request_id=request_id,
            run_id=run_id,
            lineage_id=lineage_id,
            stage_name=stage_name,
            status="inconclusive",
            hypotheses=[inconclusive_hypothesis(["valid DiagnosisRequest"])],
            leading_hypothesis_id=None,
            missing_evidence=["valid DiagnosisRequest"],
            issues=[
                ContractIssue(
                    "invalid_diagnosis_request",
                    "invalid_request",
                    "diagnose requires DiagnosisRequest",
                    blocking=True,
                )
            ],
            confidence=0.0,
            metadata={"deterministic_layer_authoritative": True},
        )


def _missing_evidence(evidence: list[NormalizedEvidence]) -> list[str]:
    kinds = {item.evidence_kind for item in evidence}
    signals = {signal for item in evidence for signal in item.signals}
    missing: list[str] = []
    if "eval_report" not in kinds:
        missing.append("eval_report")
    scorecard_present = kinds.intersection(
        {"scorecard", "deterministic_scorecard"}
    ) or any(
        isinstance(item.payload.get("deterministic_scorecard"), dict)
        and bool(item.payload.get("deterministic_scorecard"))
        for item in evidence
    )
    if not scorecard_present or "deterministic_scorecard_missing" in signals:
        missing.append("deterministic_scorecard")
    if "eval_integrity_verified" not in signals:
        missing.append("verified_eval_integrity")
    if not kinds.intersection(
        {
            "replay_verification",
            "replay_verification_report",
            "launch_config",
            "run_record",
        }
    ):
        missing.append("launch_and_replay_contract")
    if "dataset_manifest" not in kinds:
        missing.append("dataset_manifest")
    if not signals.intersection(
        {
            "tokenizer_compatible",
            "tokenizer_mismatch",
            "tokenizer_incompatible",
            "wrapper_compatible",
            "wrapper_mismatch",
        }
    ):
        missing.append("tokenizer_and_wrapper_compatibility")
    if not signals.intersection(
        {
            "checkpoint_verified",
            "checkpoint_hash_mismatch",
            "checkpoint_corrupt",
            "resume_checkpoint_mismatch",
        }
    ):
        missing.append("checkpoint_integrity")
    if not kinds.intersection(
        {"training_metrics", "training_plan", "runtime_event"}
    ) and not signals.intersection(
        {
            "optimizer_stable",
            "optimizer_pathology",
            "numerically_stable",
            "non_finite_loss",
            "non_finite_gradient",
            "training_sufficient",
            "undertraining_detected",
        }
    ):
        missing.append("training_dynamics")
    return sorted(set(missing))


def _eval_integrity_verified(evidence: list[NormalizedEvidence]) -> bool:
    verified = [item for item in evidence if "eval_integrity_verified" in item.signals]
    contradicted = [
        item
        for item in evidence
        if {
            "eval_integrity_failed",
            "eval_pack_unverified",
            "deterministic_scorecard_missing",
        }.intersection(item.signals)
    ]
    return bool(verified) and not contradicted


def _report_id(
    request: DiagnosisRequest, observations: list[EvidenceObservation]
) -> str:
    payload = "|".join(
        [request.request_id, request.run_id, request.lineage_id, request.stage_name]
        + sorted(item.observation_id for item in observations)
    )
    return f"diag-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _deduplicate_issues(issues: list[ContractIssue]) -> list[ContractIssue]:
    unique: dict[tuple[str, str], ContractIssue] = {}
    for issue in issues:
        unique[(issue.code, issue.message)] = issue
    return [unique[key] for key in sorted(unique)]
