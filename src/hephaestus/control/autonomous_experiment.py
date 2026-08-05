"""Governed integration layer for the autonomous experiment subsystems.

This module composes the independently developed services without changing the
mandatory control-spine order or granting any subsystem authority it does not
own.  It deliberately keeps approval, readiness, persistence, and judge
boundaries explicit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Protocol

from hephaestus.data.acquisition import DatasetAcquisitionApproval
from hephaestus.data.preprocessing import (
    AutonomousDataPreprocessor,
    DataFactoryResult,
)
from hephaestus.data.registry import DatasetDiscoveryResult, DatasetProviderRegistry
from hephaestus.data.selection import DeterministicDatasetSelectionService
from hephaestus.diagnosis.evidence import EvidenceAdapter
from hephaestus.diagnosis.service import EvidenceBasedDiagnosisService
from hephaestus.interfaces.discovery import (
    ModelDiscoveryProvider,
    ModelSelectionService,
)
from hephaestus.interfaces.services import (
    ExperimentEvaluationService as ExperimentEvaluationProtocol,
    TrainingLifecycleService,
)
from hephaestus.planning.service import ClosedLoopExperimentPlanner
from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.diagnosis_contract import DiagnosisReport, DiagnosisRequest
from hephaestus.schemas.discovery_contract import (
    DatasetCandidate,
    DatasetSearchRequest,
    DatasetSelectionDecision,
    ModelCandidate,
    ModelSearchRequest,
    ModelSelectionDecision,
)
from hephaestus.schemas.experiment_contract import (
    ExperimentComparison,
    ExperimentProposal,
    InterventionProposal,
    TrainingControlRequest,
    TrainingRunHandle,
)


class IntegrationRecordSink(Protocol):
    """Append-only persistence boundary for decision-critical integration records."""

    def append(self, kind: str, payload: dict[str, object]) -> None: ...


@dataclass(slots=True)
class InMemoryIntegrationRecordSink:
    """Deterministic sink for tests and local composition smoke checks."""

    records: list[dict[str, object]] = field(default_factory=list)

    def append(self, kind: str, payload: dict[str, object]) -> None:
        self.records.append({"kind": kind, **deepcopy(payload)})


_POSITIVE_SIGNAL_KEYS = frozenset(
    {
        "eval_integrity_verified",
        "reproducibility_verified",
        "data_quality_verified",
        "data_coverage_verified",
        "wrapper_compatible",
        "tokenizer_compatible",
        "architecture_compatible",
        "optimizer_stable",
        "numerically_stable",
        "training_sufficient",
        "no_overfitting",
        "decoding_verified",
        "runtime_healthy",
        "checkpoint_verified",
        "model_family_adequate",
    }
)
_TRUE_WORDS = frozenset(
    {
        "true",
        "yes",
        "passed",
        "verified",
        "compatible",
        "healthy",
        "stable",
        "sufficient",
        "adequate",
        "reproducible",
        "completed",
    }
)
_FALSE_WORDS = frozenset(
    {
        "false",
        "no",
        "failed",
        "mismatch",
        "incompatible",
        "unverified",
        "unstable",
        "insufficient",
        "missing",
        "corrupt",
    }
)


def normalize_diagnostic_truth_values(value: object) -> object:
    """Normalize positive evidence fields before legacy signal extraction.

    The diagnosis subsystem accepts heterogeneous records.  Positive fields such
    as ``tokenizer_compatible`` must interpret ``"verified"`` as true and
    ``"failed"`` as false; generic string truthiness is unsafe for those keys.
    """

    if isinstance(value, list):
        return [normalize_diagnostic_truth_values(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        child = normalize_diagnostic_truth_values(raw_value)
        if key in _POSITIVE_SIGNAL_KEYS:
            child = _positive_truth(child)
        elif key == "compatible" and isinstance(child, (bool, str)):
            child = _positive_truth(child)
        normalized[key] = child

    tokenizer = normalized.get("tokenizer_compatibility")
    if isinstance(tokenizer, dict) and isinstance(tokenizer.get("compatible"), bool):
        tokenizer["tokenizer_compatible"] = tokenizer["compatible"]
    wrapper = normalized.get("wrapper_policy") or normalized.get("wrapper")
    if isinstance(wrapper, dict) and isinstance(wrapper.get("compatible"), bool):
        wrapper["wrapper_compatible"] = wrapper["compatible"]
    return normalized


def _positive_truth(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().casefold()
        if token in _TRUE_WORDS:
            return True
        if token in _FALSE_WORDS:
            return False
    return value


@dataclass(slots=True)
class TruthNormalizingEvidenceAdapter:
    delegate: EvidenceAdapter

    def load(self, request: DiagnosisRequest) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for record in self.delegate.load(request):
            normalized = normalize_diagnostic_truth_values(dict(record))
            if isinstance(normalized, dict):
                records.append(normalized)
        return records


@dataclass(slots=True)
class IntegratedDiagnosisService:
    """Use the diagnosis implementation with key-aware evidence normalization."""

    delegate: EvidenceBasedDiagnosisService

    def diagnose(self, request: DiagnosisRequest) -> DiagnosisReport:
        normalized_failures = [
            item
            for item in (
                normalize_diagnostic_truth_values(record)
                for record in request.observed_failures
            )
            if isinstance(item, dict)
        ]
        normalized_request = replace(request, observed_failures=normalized_failures)
        service = EvidenceBasedDiagnosisService(
            evidence_adapters=tuple(
                TruthNormalizingEvidenceAdapter(adapter)
                for adapter in self.delegate.evidence_adapters
            ),
            policy=self.delegate.policy,
            explanation_adapter=self.delegate.explanation_adapter,
        )
        return service.diagnose(normalized_request)


@dataclass(slots=True)
class ApprovalAwareDatasetSelectionService:
    """Select acceptable candidates while keeping acquisition approval mandatory.

    The underlying data selector correctly identifies candidates that need
    approval, but reports them as blocked without selected IDs.  That prevents
    the planner from carrying the approval requirement forward.  This adapter
    converts only otherwise-acceptable candidates into ``selected`` decisions;
    acquisition still requires a concrete ``DatasetAcquisitionApproval``.
    """

    delegate: DeterministicDatasetSelectionService = field(
        default_factory=DeterministicDatasetSelectionService
    )

    def select(
        self,
        request: DatasetSearchRequest,
        candidates: Sequence[DatasetCandidate],
    ) -> DatasetSelectionDecision:
        decision = self.delegate.select(request, candidates)
        if decision.status != "blocked" or decision.selected_candidate_ids:
            return decision

        audits = decision.metadata.get("candidate_audits", {})
        if not isinstance(audits, dict):
            return decision
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        approval_candidates: list[tuple[str, float, list[str]]] = []
        for candidate_id in decision.ranked_candidate_ids:
            audit = audits.get(candidate_id)
            if not isinstance(audit, dict):
                continue
            rejected = audit.get("rejected_reasons", [])
            approvals = [str(item) for item in audit.get("required_approvals", [])]
            try:
                score = float(audit.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if not rejected and approvals and score >= self.delegate.minimum_score:
                approval_candidates.append((candidate_id, score, approvals))
        if not approval_candidates:
            return decision

        maximum = max(1, int(request.metadata.get("max_selected_candidates", 1) or 1))
        delta = max(0.0, float(request.metadata.get("mixture_score_delta", 0.05) or 0.05))
        best = approval_candidates[0][1]
        chosen = [item for item in approval_candidates if best - item[1] <= delta][:maximum]
        selected_ids = [item[0] for item in chosen]
        raw_weights = {candidate_id: max(score, 1e-12) for candidate_id, score, _ in chosen}
        total = sum(raw_weights.values())
        weights = {
            candidate_id: round(score / total, 12)
            for candidate_id, score in raw_weights.items()
        }
        if weights:
            last = selected_ids[-1]
            weights[last] = round(1.0 - sum(weights[item] for item in selected_ids[:-1]), 12)

        required_approvals = sorted(
            {approval for _, _, approvals in chosen for approval in approvals}
        )
        operations = sorted(
            {
                str(operation)
                for candidate_id in selected_ids
                for operation in (
                    audits[candidate_id].get("preprocessing_requirements", [])
                    if isinstance(audits.get(candidate_id), dict)
                    else []
                )
            }
        )
        rejected_candidates = dict(decision.rejected_candidates)
        for candidate_id in selected_ids:
            rejected_candidates.pop(candidate_id, None)
        evidence_refs = sorted(
            {
                *request.evidence_refs,
                *(
                    ref
                    for candidate_id in selected_ids
                    for ref in candidate_by_id[candidate_id].evidence_refs
                    if candidate_id in candidate_by_id
                ),
            }
        )
        issues = [
            issue
            for issue in decision.issues
            if issue.code != "dataset_selection_approval_required"
        ]
        issues.append(
            ContractIssue(
                code="dataset_selection_requires_acquisition_approval",
                category="approval_required",
                message="candidate selection is recorded, but acquisition remains blocked until explicit approval evidence is supplied",
                blocking=False,
                evidence_refs=evidence_refs,
                metadata={"required_approvals": required_approvals},
            )
        )
        confidence = min(0.85, max(score for _, score, _ in chosen))
        return replace(
            decision,
            status="selected",
            selected_candidate_ids=selected_ids,
            rejected_candidates=rejected_candidates,
            selection_rationale=(
                "selected highest-scoring policy-compatible candidate(s); explicit approval is still required before acquisition"
            ),
            mixture_weights=weights,
            preprocessing_requirements={"operations": operations},
            required_approvals=required_approvals,
            evidence_refs=evidence_refs,
            issues=issues,
            confidence=round(confidence, 8),
            metadata={
                **decision.metadata,
                "approval_gate": "required_before_acquisition",
                "selected_subject_to_approval": True,
            },
        )


@dataclass(slots=True)
class GuardedTrainingLifecycleService:
    """Convert corrupted persisted control evidence into structured failures."""

    delegate: TrainingLifecycleService

    def launch(self, proposal: ExperimentProposal) -> TrainingRunHandle:
        return self.delegate.launch(proposal)

    def status(self, run_id: str) -> TrainingRunHandle:
        try:
            return self.delegate.status(run_id)
        except Exception as exc:  # service boundary must not leak persistence errors
            return TrainingRunHandle(
                run_id=run_id,
                experiment_id="unknown",
                backend_id="integration_guard",
                status="failed",
                issues=[
                    ContractIssue(
                        code="training_status_evidence_invalid",
                        category="artifact_integrity",
                        message=f"training status evidence could not be read: {type(exc).__name__}",
                        retryable=True,
                        blocking=True,
                    )
                ],
            )

    def control(self, request: TrainingControlRequest) -> TrainingRunHandle:
        try:
            return self.delegate.control(request)
        except Exception as exc:  # malformed resume/job evidence is an explicit failure
            current = self.status(request.run_id)
            payload = current.to_dict()
            payload["status"] = "failed"
            payload["issues"] = [
                *payload.get("issues", []),
                ContractIssue(
                    code="training_control_evidence_invalid",
                    category="artifact_integrity",
                    message=f"training control evidence could not be validated: {type(exc).__name__}",
                    retryable=True,
                    blocking=True,
                    metadata={"action": request.action},
                ).to_dict(),
            ]
            return TrainingRunHandle.from_dict(payload)


@dataclass(frozen=True, slots=True)
class PlanningBundle:
    diagnosis: DiagnosisReport
    interventions: tuple[InterventionProposal, ...]


@dataclass(frozen=True, slots=True)
class DiscoveryBundle:
    dataset_request: DatasetSearchRequest | None = None
    dataset_discovery: DatasetDiscoveryResult | None = None
    dataset_selection: DatasetSelectionDecision | None = None
    model_request: ModelSearchRequest | None = None
    model_candidates: tuple[ModelCandidate, ...] = ()
    model_selection: ModelSelectionDecision | None = None


@dataclass(slots=True)
class AutonomousExperimentCoordinator:
    """Narrow final-integration coordinator; it does not replace the orchestrator."""

    diagnosis_service: IntegratedDiagnosisService
    planner: ClosedLoopExperimentPlanner
    dataset_registry: DatasetProviderRegistry
    dataset_selector: ApprovalAwareDatasetSelectionService
    model_providers: Mapping[str, ModelDiscoveryProvider]
    model_selector: ModelSelectionService
    training_service: GuardedTrainingLifecycleService
    evaluation_service: ExperimentEvaluationProtocol
    record_sink: IntegrationRecordSink
    data_preprocessor: AutonomousDataPreprocessor | None = None

    def diagnose_and_plan(self, request: DiagnosisRequest) -> PlanningBundle:
        diagnosis = self.diagnosis_service.diagnose(request)
        self._record("diagnosis_report", diagnosis.to_dict())
        interventions = tuple(self.planner.propose_interventions(diagnosis))
        for intervention in interventions:
            self._record("intervention_proposal", intervention.to_dict())
        return PlanningBundle(diagnosis=diagnosis, interventions=interventions)

    def discover(
        self,
        diagnosis: DiagnosisReport,
        intervention: InterventionProposal,
    ) -> DiscoveryBundle:
        dataset_request, model_request = self.planner.create_discovery_requests(
            diagnosis, intervention
        )
        dataset_discovery: DatasetDiscoveryResult | None = None
        dataset_selection: DatasetSelectionDecision | None = None
        model_candidates: tuple[ModelCandidate, ...] = ()
        model_selection: ModelSelectionDecision | None = None

        if dataset_request is not None:
            self._record("dataset_search_request", dataset_request.to_dict())
            dataset_discovery = self.dataset_registry.discover(dataset_request)
            dataset_selection = self.dataset_selector.select(
                dataset_request, dataset_discovery.candidates
            )
            combined_issues = [*dataset_selection.issues, *dataset_discovery.issues]
            status = dataset_selection.status
            if any(issue.blocking for issue in dataset_discovery.issues) and not dataset_selection.selected_candidate_ids:
                status = "blocked"
            dataset_selection = replace(
                dataset_selection, issues=combined_issues, status=status
            )
            self._record("dataset_selection_decision", dataset_selection.to_dict())

        if model_request is not None:
            model_request, request_issues = self._normalize_model_request(
                diagnosis, model_request
            )
            self._record("model_search_request", model_request.to_dict())
            candidates, provider_issues = self._discover_models(model_request)
            model_candidates = tuple(candidates)
            model_selection = self.model_selector.select(model_request, candidates)
            combined = [*model_selection.issues, *request_issues, *provider_issues]
            status = model_selection.status
            if any(issue.blocking for issue in combined) and not model_selection.selected_candidate_id:
                status = "blocked"
            confidence = model_selection.confidence
            if request_issues:
                confidence = min(confidence, 0.55)
            model_selection = replace(
                model_selection,
                issues=combined,
                status=status,
                confidence=confidence,
            )
            self._record("model_selection_decision", model_selection.to_dict())

        return DiscoveryBundle(
            dataset_request=dataset_request,
            dataset_discovery=dataset_discovery,
            dataset_selection=dataset_selection,
            model_request=model_request,
            model_candidates=model_candidates,
            model_selection=model_selection,
        )

    def build_experiment(
        self,
        diagnosis: DiagnosisReport,
        intervention: InterventionProposal,
        discovery: DiscoveryBundle,
    ) -> ExperimentProposal:
        proposal = self.planner.propose_experiment(
            diagnosis,
            intervention,
            discovery.dataset_selection,
            discovery.model_selection,
        )
        self._record("experiment_proposal", proposal.to_dict())
        return proposal

    def prepare_selected_dataset(
        self,
        *,
        proposal: ExperimentProposal,
        discovery: DiscoveryBundle,
        approval: DatasetAcquisitionApproval,
        candidates: Sequence[DatasetCandidate],
        tokenizer_ref: str | None = None,
    ) -> DataFactoryResult:
        if self.data_preprocessor is None:
            raise RuntimeError("data preprocessor is not configured")
        selection = discovery.dataset_selection
        if selection is None or len(selection.selected_candidate_ids) != 1:
            raise ValueError("exactly one selected dataset candidate is required for materialization")
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        candidate_id = selection.selected_candidate_ids[0]
        if candidate_id not in candidate_by_id:
            raise ValueError("selected dataset candidate payload is missing")
        result = self.data_preprocessor.process(
            run_id=proposal.run_id,
            lineage_id=proposal.lineage_id,
            stage_name=proposal.stage_name,
            candidate=candidate_by_id[candidate_id],
            selection=selection,
            approval=approval,
            tokenizer_ref=tokenizer_ref,
        )
        self._record("dataset_manifest", result.manifest.to_dict())
        self._record("preprocessing_report", result.preprocessing_report.to_dict())
        self._record("trainable_data_contract", result.trainable_data_contract.to_dict())
        return result

    def bind_training_inputs(
        self,
        proposal: ExperimentProposal,
        data_result: DataFactoryResult,
        model_candidate: ModelCandidate,
        *,
        training_recipe_ref: str,
        max_steps: int,
        learning_rate: float = 0.05,
        backend_id: str = "local_fixture",
    ) -> ExperimentProposal:
        processed_ref = str(data_result.processing_evidence["processed_dataset_ref"])
        constraints = {
            **proposal.training_constraints,
            "backend_id": backend_id,
            "model_id": model_candidate.model_id,
            "model_revision": model_candidate.revision or "",
            "architecture_family": model_candidate.architecture_family,
            "tokenizer_ref": data_result.manifest.tokenizer_ref
            or model_candidate.tokenizer_ref
            or "",
            "training_recipe_ref": training_recipe_ref,
            # The local smoke trainer consumes the processed artifact itself.
            "data_contract_ref": processed_ref,
            "data_contract_hash": data_result.processed_content_hash.removeprefix("sha256:"),
            "max_steps": max_steps,
            "learning_rate": learning_rate,
        }
        bound = replace(proposal, training_constraints=constraints)
        self._record("experiment_training_binding", bound.to_dict())
        return bound

    def launch_approved(
        self,
        proposal: ExperimentProposal,
        approval_evidence: Mapping[str, str],
    ) -> TrainingRunHandle:
        missing = sorted(set(proposal.required_approvals) - set(approval_evidence))
        if proposal.status not in {"ready", "approved"} or missing:
            issue = ContractIssue(
                code="experiment_launch_not_authorized",
                category="approval_required" if missing else "policy_blocked",
                message=(
                    f"missing approval evidence: {', '.join(missing)}"
                    if missing
                    else "experiment must be ready or approved before launch"
                ),
                blocking=True,
                evidence_refs=sorted(approval_evidence.values()),
            )
            handle = TrainingRunHandle(
                run_id=proposal.run_id,
                experiment_id=proposal.experiment_id,
                backend_id="not_launched",
                status="failed",
                issues=[issue],
                metadata={"launch_attempted": False},
            )
            self._record("training_run_handle", handle.to_dict())
            return handle
        approved = replace(
            proposal,
            status="approved",
            metadata={
                **proposal.metadata,
                "approval_evidence": dict(approval_evidence),
            },
        )
        handle = self.training_service.launch(approved)
        self._record("training_run_handle", handle.to_dict())
        return handle

    def compare(
        self,
        proposal: ExperimentProposal,
        runs: Sequence[TrainingRunHandle],
        *,
        baseline_resolver: Callable[[str], TrainingRunHandle | None] | None = None,
    ) -> ExperimentComparison:
        run_list = list(runs)
        resolved_proposal = proposal
        baseline_ref = str(proposal.baseline_ref or "").strip()
        run_ids = {run.run_id for run in run_list}
        if baseline_ref and baseline_ref not in run_ids and baseline_resolver is not None:
            baseline = baseline_resolver(baseline_ref)
            if baseline is not None:
                if baseline.run_id not in run_ids:
                    run_list.append(baseline)
                resolved_proposal = replace(
                    proposal,
                    baseline_ref=baseline.run_id,
                    metadata={
                        **proposal.metadata,
                        "baseline_source_ref": baseline_ref,
                        "baseline_resolved_run_id": baseline.run_id,
                    },
                )
        comparison = self.evaluation_service.compare(resolved_proposal, run_list)
        self._record("experiment_comparison", comparison.to_dict())
        return comparison

    def _normalize_model_request(
        self,
        diagnosis: DiagnosisReport,
        request: ModelSearchRequest,
    ) -> tuple[ModelSearchRequest, list[ContractIssue]]:
        structured = diagnosis.metadata.get("task_requirements")
        if not isinstance(structured, list):
            structured = diagnosis.metadata.get("capability_targets")
        if isinstance(structured, list) and structured:
            return replace(request, task_requirements=[str(item) for item in structured]), []
        if request.task_requirements == [request.problem_statement]:
            return replace(request, task_requirements=[]), [
                ContractIssue(
                    code="model_task_requirements_unstructured",
                    category="missing_evidence",
                    message="model discovery proceeded without normalized task requirements; selection confidence is capped",
                    retryable=True,
                    blocking=False,
                    evidence_refs=list(request.evidence_refs),
                )
            ]
        return request, []

    def _discover_models(
        self, request: ModelSearchRequest
    ) -> tuple[list[ModelCandidate], list[ContractIssue]]:
        requested = {
            item.strip().casefold()
            for item in request.provider_allowlist
            if item.strip()
        }
        issues: list[ContractIssue] = []
        if not requested:
            return [], [
                ContractIssue(
                    code="model_provider_allowlist_required",
                    category="policy_blocked",
                    message="model discovery requires an explicit provider allowlist",
                    blocking=True,
                )
            ]
        candidates: dict[str, ModelCandidate] = {}
        for provider_id in sorted(requested):
            provider = self.model_providers.get(provider_id)
            if provider is None:
                issues.append(
                    ContractIssue(
                        code="model_provider_not_registered",
                        category="provider_unavailable",
                        message=f"requested model provider is not registered: {provider_id}",
                        blocking=False,
                    )
                )
                continue
            try:
                for candidate in provider.search(request):
                    if candidate.candidate_id in candidates:
                        issues.append(
                            ContractIssue(
                                code="duplicate_model_candidate_id",
                                category="internal_contract_violation",
                                message=f"duplicate model candidate ignored: {candidate.candidate_id}",
                            )
                        )
                        continue
                    candidates[candidate.candidate_id] = candidate
            except Exception as exc:
                issues.append(
                    ContractIssue(
                        code="model_provider_failure",
                        category="provider_unavailable",
                        message=f"model provider {provider_id} failed: {type(exc).__name__}",
                        retryable=True,
                        blocking=False,
                    )
                )
        if not candidates:
            issues.append(
                ContractIssue(
                    code="no_model_candidates",
                    category="candidate_not_found",
                    message="no model candidates were discovered",
                    retryable=True,
                    blocking=True,
                )
            )
        return [candidates[key] for key in sorted(candidates)], issues

    def _record(self, kind: str, payload: dict[str, object]) -> None:
        self.record_sink.append(kind, payload)
