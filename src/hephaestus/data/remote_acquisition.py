"""Governed planning and execution service for remote dataset acquisition."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from hephaestus.infrastructure.secrets import SecretReference, SecretsProvider
from hephaestus.providers.datasets.acquisition import (
    DatasetProviderAcquisitionError,
    ProviderDatasetFile,
    RemoteDatasetAcquisitionProvider,
)
from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.discovery_contract import (
    DatasetCandidate,
    DatasetSelectionDecision,
)
from hephaestus.storage.base import ArtifactStore

from .acquisition import DatasetAcquisitionApproval
from .acquisition_cache import DatasetAcquisitionCache, validate_relative_file_path
from .acquisition_models import (
    AcquiredFileEvidence,
    AcquisitionPlanningResult,
    AcquisitionReceipt,
    RemoteAcquisitionLimits,
    RemoteAcquisitionPlan,
    RemoteAcquisitionResult,
    TransferAttempt,
)
from .acquisition_transfer import AcquisitionFileTransfer
from .acquisition_transport import DownloadTransport, UrllibDownloadTransport


@dataclass(slots=True)
class RemoteDatasetAcquisitionService:
    """Keep planning, approval, transfer, cache, and receipts separately auditable."""

    providers: dict[str, RemoteDatasetAcquisitionProvider]
    cache: DatasetAcquisitionCache
    transport: DownloadTransport = field(default_factory=UrllibDownloadTransport)
    secrets_provider: SecretsProvider | None = None
    artifact_store: ArtifactStore | None = None
    clock: Callable[[], datetime] = field(
        default=lambda: datetime.now(timezone.utc), repr=False
    )

    def plan(
        self,
        *,
        candidate: DatasetCandidate,
        selection: DatasetSelectionDecision,
        limits: RemoteAcquisitionLimits | None = None,
        authentication_reference: SecretReference | None = None,
    ) -> AcquisitionPlanningResult:
        limits = limits or RemoteAcquisitionLimits()
        issues = self._selection_issues(candidate, selection)
        provider = self.providers.get(candidate.provider_id)
        if provider is None:
            issues.append(
                ContractIssue(
                    code="dataset_acquisition_provider_unavailable",
                    category="provider_unavailable",
                    message=f"no acquisition provider is registered for {candidate.provider_id}",
                    blocking=True,
                )
            )
        if issues:
            return AcquisitionPlanningResult("blocked", issues=tuple(issues))

        token, auth_issue = self._resolve_token(authentication_reference)
        if auth_issue is not None:
            return AcquisitionPlanningResult("blocked", issues=(auth_issue,))
        requested_revision = candidate.revision or str(
            candidate.metadata.get("requested_revision") or "main"
        )
        try:
            assert provider is not None
            snapshot = provider.resolve_revision(
                candidate.dataset_id, requested_revision, token=token
            )
            enumerated = tuple(provider.enumerate_files(snapshot, token=token))
        except DatasetProviderAcquisitionError as exc:
            return AcquisitionPlanningResult(
                "blocked", issues=(exc.to_issue(evidence_refs=candidate.evidence_refs),)
            )
        except Exception as exc:  # noqa: BLE001 - provider failure boundary
            issue = ContractIssue(
                code="dataset_acquisition_provider_failure",
                category="provider_unavailable",
                message=f"dataset acquisition provider failed with {type(exc).__name__}",
                retryable=True,
                blocking=True,
                evidence_refs=list(candidate.evidence_refs),
                metadata={
                    "provider_id": candidate.provider_id,
                    "exception_type": type(exc).__name__,
                },
            )
            return AcquisitionPlanningResult("blocked", issues=(issue,))

        if (
            bool(candidate.compatibility.get("remote_code_required"))
            or snapshot.remote_code_required
        ):
            return AcquisitionPlanningResult(
                "blocked",
                issues=(
                    ContractIssue(
                        code="unsupported_remote_code",
                        category="unsupported_capability",
                        message="dataset requires remote code, which is prohibited by acquisition policy",
                        blocking=True,
                        evidence_refs=list(candidate.evidence_refs),
                    ),
                ),
            )
        if (snapshot.gated or snapshot.private) and authentication_reference is None:
            return AcquisitionPlanningResult(
                "blocked",
                issues=(
                    ContractIssue(
                        code="authentication_reference_required",
                        category="approval_required",
                        message="gated or private dataset acquisition requires an injected secret reference",
                        blocking=True,
                    ),
                ),
            )

        selected_paths = (
            {
                str(item)
                for item in candidate.metadata.get("acquisition_files", [])
                if str(item).strip()
            }
            if isinstance(candidate.metadata.get("acquisition_files"), list)
            else set()
        )
        files: list[ProviderDatasetFile] = []
        warnings: list[str] = []
        path_issues: list[ContractIssue] = []
        for provider_file in enumerated:
            try:
                path = validate_relative_file_path(provider_file.relative_path)
            except ValueError:
                path_issues.append(
                    ContractIssue(
                        code="unsafe_provider_path",
                        category="artifact_integrity",
                        message="provider returned an unsafe file path",
                        blocking=True,
                        metadata={"provider_id": candidate.provider_id},
                    )
                )
                continue
            if path.casefold().endswith(
                (".py", ".sh", ".exe", ".dll", ".so", ".dylib")
            ):
                if path in selected_paths:
                    path_issues.append(
                        ContractIssue(
                            code="unsupported_remote_code",
                            category="unsupported_capability",
                            message="executable repository files cannot be acquired as dataset material",
                            blocking=True,
                            metadata={"relative_path": path},
                        )
                    )
                continue
            wanted = (
                path in selected_paths
                if selected_paths
                else path.casefold().endswith(limits.allowed_suffixes)
            )
            if wanted:
                files.append(provider_file)
        if path_issues:
            return AcquisitionPlanningResult("blocked", issues=tuple(path_issues))
        if selected_paths - {provider_file.relative_path for provider_file in files}:
            return AcquisitionPlanningResult(
                "blocked",
                issues=(
                    ContractIssue(
                        code="requested_dataset_files_missing",
                        category="candidate_not_found",
                        message="one or more explicitly requested dataset files were absent or prohibited",
                        blocking=True,
                    ),
                ),
            )
        if not files:
            return AcquisitionPlanningResult(
                "blocked",
                issues=(
                    ContractIssue(
                        code="no_acquirable_dataset_files",
                        category="candidate_not_found",
                        message="provider revision contains no allowlisted dataset files",
                        blocking=True,
                    ),
                ),
            )
        files.sort(key=lambda item: item.relative_path)
        if len(files) > limits.max_files:
            return AcquisitionPlanningResult(
                "blocked",
                issues=(
                    ContractIssue(
                        code="maximum_file_count_exceeded",
                        category="budget_exceeded",
                        message=f"acquisition requires {len(files)} files but the limit is {limits.max_files}",
                        blocking=True,
                    ),
                ),
            )
        known_bytes = sum(provider_file.size_bytes or 0 for provider_file in files)
        if (
            all(provider_file.size_bytes is not None for provider_file in files)
            and known_bytes > limits.max_bytes
        ):
            return AcquisitionPlanningResult(
                "blocked",
                issues=(
                    ContractIssue(
                        code="maximum_byte_budget_exceeded",
                        category="budget_exceeded",
                        message=f"acquisition requires {known_bytes} bytes but the limit is {limits.max_bytes}",
                        blocking=True,
                    ),
                ),
            )
        if any(provider_file.size_bytes is None for provider_file in files):
            warnings.append("one_or_more_remote_file_sizes_unknown")
        if any(provider_file.provider_hash is None for provider_file in files):
            warnings.append("one_or_more_provider_hashes_missing")
        required_approvals = set(selection.required_approvals)
        if not snapshot.license:
            required_approvals.add(f"unknown_license:{candidate.candidate_id}")
        if "provenance" in snapshot.missing_metadata:
            required_approvals.add(f"unknown_provenance:{candidate.candidate_id}")
        plan = RemoteAcquisitionPlan.create(
            selection_decision_id=selection.decision_id,
            candidate_id=candidate.candidate_id,
            snapshot=snapshot,
            files=tuple(files),
            limits=limits,
            required_approvals=tuple(sorted(required_approvals)),
            authentication_reference=(
                authentication_reference.to_dict() if authentication_reference else None
            ),
            warnings=tuple(sorted(set(warnings))),
        )
        return AcquisitionPlanningResult("ready", plan=plan)

    def acquire(
        self,
        plan: RemoteAcquisitionPlan,
        approval: DatasetAcquisitionApproval,
        *,
        authentication_reference: SecretReference | None = None,
        offline: bool = False,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> RemoteAcquisitionResult:
        started_at = self.clock().isoformat()
        approval_refs = tuple(
            sorted({ref.strip() for ref in approval.approval_refs if ref.strip()})
        )
        issues = self._approval_issues(plan, approval, approval_refs)
        early_cleanup: list[dict[str, object]] = []
        token: str | None = None
        if not issues:
            token, auth_issue = self._resolve_token(authentication_reference)
            if auth_issue is not None:
                issues.append(auth_issue)
        runtime_auth_ref = (
            authentication_reference.to_dict() if authentication_reference else None
        )
        if not issues and plan.authentication_reference != runtime_auth_ref:
            issues.append(
                ContractIssue(
                    code="authentication_reference_mismatch",
                    category="artifact_integrity",
                    message="runtime authentication reference does not match the acquisition plan",
                    blocking=True,
                )
            )
        provider = self.providers.get(plan.snapshot.provider_id)
        if not issues and provider is None:
            issues.append(
                ContractIssue(
                    code="dataset_acquisition_provider_unavailable",
                    category="provider_unavailable",
                    message="planned acquisition provider is not registered",
                    blocking=True,
                )
            )
        if not issues and not offline:
            try:
                assert provider is not None
                if not provider.revision_is_current(plan.snapshot, token=token):
                    for provider_file in plan.files:
                        key = self.cache.cache_key(
                            provider_id=plan.snapshot.provider_id,
                            dataset_id=plan.snapshot.dataset_id,
                            resolved_revision=plan.snapshot.resolved_revision,
                            file=provider_file,
                        )
                        early_cleanup.append(self.cache.clear_partial_by_key(key))
                    issues.append(
                        ContractIssue(
                            code="revision_changed",
                            category="artifact_integrity",
                            message="requested provider revision no longer resolves to the planned immutable commit",
                            blocking=True,
                        )
                    )
            except DatasetProviderAcquisitionError as exc:
                issues.append(exc.to_issue())
            except Exception as exc:  # noqa: BLE001 - provider verification boundary
                issues.append(
                    ContractIssue(
                        code="dataset_acquisition_provider_failure",
                        category="provider_unavailable",
                        message=f"provider revision verification failed with {type(exc).__name__}",
                        retryable=True,
                        blocking=True,
                        metadata={"exception_type": type(exc).__name__},
                    )
                )
        if issues:
            return RemoteAcquisitionResult(
                self._receipt(
                    plan,
                    approval_refs,
                    issues=issues,
                    cleanup=early_cleanup,
                    completion_status="failed",
                    started_at=started_at,
                )
            )

        acquired: list[AcquiredFileEvidence] = []
        attempts: list[TransferAttempt] = []
        recoveries: list[dict[str, object]] = []
        cleanup: list[dict[str, object]] = []
        warnings = list(plan.warnings)
        transferred_bytes = 0
        failure_status = "failed"
        transfer = AcquisitionFileTransfer(
            self.cache, self.transport, self.artifact_store
        )
        for file_index, provider_file in enumerate(plan.files, start=1):
            if cancellation_requested is not None and cancellation_requested():
                issues.append(
                    ContractIssue(
                        code="acquisition_cancelled",
                        category="runtime_failure",
                        message="dataset acquisition was cancelled before the next file transfer",
                        retryable=True,
                        blocking=True,
                    )
                )
                failure_status = "cancelled"
                break
            lookup = self.cache.lookup(
                provider_id=plan.snapshot.provider_id,
                dataset_id=plan.snapshot.dataset_id,
                resolved_revision=plan.snapshot.resolved_revision,
                file=provider_file,
            )
            if (
                lookup.status == "hit"
                and lookup.path is not None
                and lookup.content_hash is not None
            ):
                artifact_ref, artifact_hash, artifact_issue = transfer.store_artifact(
                    lookup.path, lookup.content_hash
                )
                if artifact_issue:
                    issues.append(artifact_issue)
                    break
                acquired.append(
                    AcquiredFileEvidence(
                        relative_path=provider_file.relative_path,
                        source_url=provider_file.source_url,
                        provider_object_id=provider_file.object_id,
                        size_bytes=int(lookup.byte_size or 0),
                        provider_declared_hash=provider_file.provider_hash,
                        provider_hash_algorithm=provider_file.provider_hash_algorithm,
                        provider_hash_status="cache_reuse_verified",
                        transport_checksum=None,
                        transport_checksum_status="not_applicable_cache_reuse",
                        local_content_hash=lookup.content_hash,
                        cache_key=lookup.cache_key,
                        cache_status="hit_verified",
                        cache_ref=str(lookup.path),
                        artifact_ref=artifact_ref,
                        artifact_store_content_hash=artifact_hash,
                    )
                )
                attempts.append(
                    TransferAttempt(
                        provider_file.relative_path,
                        file_index,
                        "cache_reuse",
                        "completed",
                    )
                )
                continue
            if lookup.status == "corrupt":
                warnings.append(f"corrupt_cache_rejected:{provider_file.relative_path}")
                if not offline:
                    cleanup.append(self.cache.quarantine_corrupt(lookup.cache_key))
            if offline:
                issues.append(
                    ContractIssue(
                        code="offline_artifact_missing",
                        category="missing_evidence",
                        message=f"verified cache content is unavailable for {provider_file.relative_path}",
                        blocking=True,
                    )
                )
                break
            outcome = transfer.transfer_file(
                plan,
                provider_file,
                attempt_number=file_index,
                token=token,
                acquired_bytes=sum(item.size_bytes for item in acquired),
                cancellation_requested=cancellation_requested,
            )
            attempts.append(outcome.attempt)
            recoveries.extend(outcome.recovery)
            cleanup.extend(outcome.cleanup)
            transferred_bytes += outcome.transferred_bytes
            if outcome.issue is not None:
                issues.append(outcome.issue)
                if outcome.issue.code == "acquisition_cancelled":
                    failure_status = "cancelled"
                break
            assert outcome.evidence is not None
            acquired.append(outcome.evidence)

        completion_status = (
            "completed"
            if len(acquired) == len(plan.files) and not issues
            else failure_status
        )
        if acquired and completion_status == "failed":
            completion_status = "partial"
        return RemoteAcquisitionResult(
            self._receipt(
                plan,
                approval_refs,
                acquired=acquired,
                attempts=attempts,
                recoveries=recoveries,
                cleanup=cleanup,
                issues=issues,
                warnings=warnings,
                transferred_bytes=transferred_bytes,
                completion_status=completion_status,
                started_at=started_at,
            )
        )

    def _receipt(
        self,
        plan: RemoteAcquisitionPlan,
        approval_refs: tuple[str, ...],
        *,
        acquired: Sequence[AcquiredFileEvidence] = (),
        attempts: Sequence[TransferAttempt] = (),
        recoveries: Sequence[dict[str, object]] = (),
        cleanup: Sequence[dict[str, object]] = (),
        issues: Sequence[ContractIssue] = (),
        warnings: Sequence[str] = (),
        transferred_bytes: int = 0,
        completion_status: str,
        started_at: str,
    ) -> AcquisitionReceipt:
        cache_states = {item.cache_status for item in acquired}
        cache_status = (
            "all_hits"
            if cache_states == {"hit_verified"}
            else "mixed"
            if "hit_verified" in cache_states
            else "misses_stored"
            if acquired
            else "no_complete_files"
        )
        missing = set(plan.snapshot.missing_metadata)
        for provider_file in plan.files:
            if provider_file.provider_hash is None:
                missing.add(f"provider_hash:{provider_file.relative_path}")
        for item in acquired:
            if item.transport_checksum is None:
                missing.add(f"transport_checksum:{item.relative_path}")
        artifact_refs = tuple(
            sorted({item.artifact_ref for item in acquired if item.artifact_ref})
        )
        return AcquisitionReceipt.create(
            plan_id=plan.plan_id,
            selection_decision_id=plan.selection_decision_id,
            approval_refs=approval_refs,
            candidate_id=plan.candidate_id,
            provider_id=plan.snapshot.provider_id,
            dataset_id=plan.snapshot.dataset_id,
            requested_revision=plan.snapshot.requested_revision,
            resolved_revision=plan.snapshot.resolved_revision,
            acquired_files=tuple(acquired),
            byte_totals={
                "acquired": sum(item.size_bytes for item in acquired),
                "transferred": transferred_bytes,
                "planned": plan.estimated_total_bytes or 0,
            },
            cache_status=cache_status,
            dataset_card_ref=plan.snapshot.dataset_card_ref,
            dataset_card_revision=plan.snapshot.dataset_card_revision,
            license=plan.snapshot.license,
            license_source=plan.snapshot.license_source,
            transfer_attempts=tuple(attempts),
            partial_recovery_evidence=tuple(dict(item) for item in recoveries),
            artifact_refs=artifact_refs,
            warnings=tuple(sorted(set(warnings))),
            missing_evidence=tuple(sorted(missing)),
            cleanup=tuple(dict(item) for item in cleanup),
            completion_status=completion_status,
            issues=tuple(issues),
            observations={
                "started_at": started_at,
                "completed_at": self.clock().isoformat(),
            },
        )

    @staticmethod
    def _selection_issues(
        candidate: DatasetCandidate, selection: DatasetSelectionDecision
    ) -> list[ContractIssue]:
        if (
            selection.status != "selected"
            or candidate.candidate_id not in selection.selected_candidate_ids
        ):
            return [
                ContractIssue(
                    code="candidate_not_selected",
                    category="policy_blocked",
                    message="remote acquisition planning requires a selected candidate",
                    blocking=True,
                )
            ]
        if candidate.provider_id == "local_fixture":
            return [
                ContractIssue(
                    code="remote_provider_required",
                    category="unsupported_capability",
                    message="remote acquisition service does not replace local fixture acquisition",
                    blocking=True,
                )
            ]
        return []

    @staticmethod
    def _approval_issues(
        plan: RemoteAcquisitionPlan,
        approval: DatasetAcquisitionApproval,
        approval_refs: tuple[str, ...],
    ) -> list[ContractIssue]:
        if (
            approval.selection_decision_id != plan.selection_decision_id
            or plan.candidate_id not in approval.approved_candidate_ids
            or not approval_refs
        ):
            return [
                ContractIssue(
                    code="dataset_acquisition_approval_missing",
                    category="approval_required",
                    message="explicit approval for this selection decision and candidate is required",
                    blocking=True,
                )
            ]
        approved_requirements = {
            str(item).strip()
            for item in approval.approved_requirements
            if str(item).strip()
        }
        missing_requirements = sorted(
            set(plan.required_approvals) - approved_requirements
        )
        if missing_requirements:
            return [
                ContractIssue(
                    code="dataset_policy_approvals_missing",
                    category="approval_required",
                    message="acquisition approval does not cover every policy requirement in the plan",
                    blocking=True,
                    metadata={"missing_requirements": missing_requirements},
                )
            ]
        return []

    def _resolve_token(
        self, reference: SecretReference | None
    ) -> tuple[str | None, ContractIssue | None]:
        if reference is None:
            return None, None
        if self.secrets_provider is None:
            return None, ContractIssue(
                code="secret_provider_unavailable",
                category="provider_unavailable",
                message="authentication was requested but no secrets provider was injected",
                blocking=True,
                metadata={"secret_reference": reference.to_dict()},
            )
        try:
            return self.secrets_provider.resolve(reference), None
        except Exception as exc:  # noqa: BLE001 - injected secret-provider boundary
            return None, ContractIssue(
                code="authentication_failure",
                category="provider_unavailable",
                message=f"secret reference resolution failed with {type(exc).__name__}",
                blocking=True,
                metadata={
                    "secret_reference": reference.to_dict(),
                    "exception_type": type(exc).__name__,
                },
            )


__all__ = ["RemoteDatasetAcquisitionService"]
