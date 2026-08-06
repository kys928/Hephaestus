from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from hephaestus.data import (
    AcquisitionReceipt,
    DatasetAcquisitionApproval,
    DatasetAcquisitionCache,
    RemoteAcquisitionLimits,
    RemoteDatasetAcquisitionService,
)
from hephaestus.infrastructure.secrets import (
    EnvironmentSecretsProvider,
    SecretReference,
)
from hephaestus.providers.datasets import (
    DatasetProviderAcquisitionError,
    ProviderDatasetFile,
    ProviderDatasetSnapshot,
)
from hephaestus.schemas.discovery_contract import (
    DatasetCandidate,
    DatasetSelectionDecision,
)
from hephaestus.storage import FileSystemArtifactStore

REVISION = "a" * 40
URL = f"https://provider.invalid/datasets/example/resolve/{REVISION}/data/train.jsonl"
DATA = b'{"text":"one"}\n{"text":"two"}\n'


@dataclass(slots=True)
class FakeRemoteProvider:
    snapshot: ProviderDatasetSnapshot
    files: tuple[ProviderDatasetFile, ...]
    provider_id: str = "fake_remote"
    current: bool = True
    failure: DatasetProviderAcquisitionError | None = None
    tokens: list[str | None] = field(default_factory=list)

    def resolve_revision(
        self, dataset_id: str, requested_revision: str, *, token: str | None = None
    ):
        self.tokens.append(token)
        if self.failure:
            raise self.failure
        assert dataset_id == self.snapshot.dataset_id
        return replace(self.snapshot, requested_revision=requested_revision)

    def enumerate_files(
        self, snapshot: ProviderDatasetSnapshot, *, token: str | None = None
    ):
        self.tokens.append(token)
        return self.files

    def revision_is_current(
        self, snapshot: ProviderDatasetSnapshot, *, token: str | None = None
    ):
        self.tokens.append(token)
        return self.current


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int,
        headers: Mapping[str, str],
        fail_after: int | None = None,
    ) -> None:
        self.body = body
        self.status = status
        self.headers = dict(headers)
        self.fail_after = fail_after
        self.position = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if self.fail_after is not None and self.position >= self.fail_after:
            raise OSError("simulated interruption")
        remaining = len(self.body) - self.position
        if remaining <= 0:
            return b""
        wanted = remaining if size < 0 else min(size, remaining)
        if self.fail_after is not None:
            wanted = min(wanted, self.fail_after - self.position)
        chunk = self.body[self.position : self.position + wanted]
        self.position += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class FakeTransport:
    payloads: dict[str, bytes]
    support_range: bool = True
    fail_after: int | None = None
    include_transport_checksum: bool = True
    requests: list[dict[str, object]] = field(default_factory=list)

    def open(self, url: str, *, headers: Mapping[str, str], timeout_seconds: float):
        self.requests.append(
            {"url": url, "headers": dict(headers), "timeout": timeout_seconds}
        )
        body = self.payloads[url]
        offset = 0
        range_header = headers.get("Range")
        response_headers: dict[str, str] = {"ETag": '"immutable-etag"'}
        if self.include_transport_checksum:
            response_headers["X-Checksum-Sha256"] = hashlib.sha256(body).hexdigest()
        if range_header and self.support_range:
            offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
            response_headers["Content-Range"] = (
                f"bytes {offset}-{len(body) - 1}/{len(body)}"
            )
            status = 206
            response_body = body[offset:]
        else:
            status = 200
            response_body = body
        fail_after = self.fail_after
        self.fail_after = None
        return FakeResponse(
            response_body,
            status=status,
            headers=response_headers,
            fail_after=fail_after,
        )


def _snapshot(**overrides: object) -> ProviderDatasetSnapshot:
    values: dict[str, object] = {
        "provider_id": "fake_remote",
        "dataset_id": "org/example",
        "requested_revision": "main",
        "resolved_revision": REVISION,
        "dataset_card_ref": f"https://provider.invalid/datasets/org/example/blob/{REVISION}/README.md",
        "dataset_card_revision": REVISION,
        "license": "mit",
        "license_source": "provider_dataset_card",
        "citation": "Fixture citation",
        "authors": ("Fixture Author",),
    }
    values.update(overrides)
    return ProviderDatasetSnapshot(**values)


def _file(payload: bytes = DATA, **overrides: object) -> ProviderDatasetFile:
    values: dict[str, object] = {
        "relative_path": "data/train.jsonl",
        "source_url": URL,
        "size_bytes": len(payload),
        "provider_hash": hashlib.sha256(payload).hexdigest(),
        "provider_hash_algorithm": "sha256",
        "etag": '"immutable-etag"',
        "object_id": "provider-object-1",
    }
    values.update(overrides)
    return ProviderDatasetFile(**values)


def _candidate(**overrides: object) -> DatasetCandidate:
    values: dict[str, object] = {
        "candidate_id": "candidate-remote-1",
        "provider_id": "fake_remote",
        "dataset_id": "org/example",
        "revision": "main",
        "license": "mit",
        "provenance": {"provider": "fake_remote"},
        "compatibility": {"remote_code_required": False},
        "evidence_refs": ["provider://org/example"],
    }
    values.update(overrides)
    return DatasetCandidate(**values)


def _selection(
    candidate: DatasetCandidate, **overrides: object
) -> DatasetSelectionDecision:
    values: dict[str, object] = {
        "decision_id": "selection-remote-1",
        "request_id": "request-remote-1",
        "status": "selected",
        "selected_candidate_ids": [candidate.candidate_id],
        "ranked_candidate_ids": [candidate.candidate_id],
    }
    values.update(overrides)
    return DatasetSelectionDecision(**values)


def _approval(candidate: DatasetCandidate, selection: DatasetSelectionDecision):
    return DatasetAcquisitionApproval(
        selection.decision_id,
        (candidate.candidate_id,),
        ("approval://operator/dataset-1",),
    )


def _service(
    tmp_path: Path,
    *,
    provider: FakeRemoteProvider | None = None,
    transport: FakeTransport | None = None,
    secret_value: str | None = None,
    artifact_store: bool = False,
) -> tuple[RemoteDatasetAcquisitionService, FakeRemoteProvider, FakeTransport]:
    provider = provider or FakeRemoteProvider(_snapshot(), (_file(),))
    transport = transport or FakeTransport({URL: DATA})
    service = RemoteDatasetAcquisitionService(
        providers={provider.provider_id: provider},
        cache=DatasetAcquisitionCache(tmp_path / "cache"),
        transport=transport,
        secrets_provider=(
            EnvironmentSecretsProvider(environ={"HF_DATASET_TOKEN": secret_value})
            if secret_value is not None
            else None
        ),
        artifact_store=FileSystemArtifactStore(tmp_path / "artifacts")
        if artifact_store
        else None,
    )
    return service, provider, transport


def _plan(
    service: RemoteDatasetAcquisitionService,
    candidate: DatasetCandidate,
    selection: DatasetSelectionDecision,
    **kwargs: object,
):
    result = service.plan(candidate=candidate, selection=selection, **kwargs)
    assert result.status == "ready", [issue.to_dict() for issue in result.issues]
    assert result.plan is not None
    return result.plan


def test_floating_revision_resolves_to_immutable_deterministic_plan(
    tmp_path: Path,
) -> None:
    service, _, _ = _service(tmp_path)
    candidate = _candidate()
    selection = _selection(candidate)

    first = _plan(service, candidate, selection)
    second = _plan(service, candidate, selection)

    assert first.snapshot.requested_revision == "main"
    assert first.snapshot.resolved_revision == REVISION
    assert first.plan_id == second.plan_id
    assert first.files[0].source_url.endswith(f"/{REVISION}/data/train.jsonl")


def test_unresolved_revision_and_provider_failure_become_contract_issues(
    tmp_path: Path,
) -> None:
    failure = DatasetProviderAcquisitionError(
        "immutable_revision_unresolved",
        "provider did not return a commit",
        category="missing_evidence",
    )
    provider = FakeRemoteProvider(_snapshot(), (_file(),), failure=failure)
    service, _, _ = _service(tmp_path, provider=provider)
    candidate = _candidate()

    result = service.plan(candidate=candidate, selection=_selection(candidate))

    assert result.status == "blocked"
    assert result.issues[0].code == "immutable_revision_unresolved"


def test_remote_code_and_path_traversal_are_blocked(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    remote_code = _candidate(compatibility={"remote_code_required": True})
    result = service.plan(candidate=remote_code, selection=_selection(remote_code))
    assert result.issues[0].code == "unsupported_remote_code"

    provider = FakeRemoteProvider(
        _snapshot(), (_file(relative_path="../escape.jsonl"),)
    )
    service, _, _ = _service(tmp_path, provider=provider)
    candidate = _candidate()
    unsafe = service.plan(candidate=candidate, selection=_selection(candidate))
    assert unsafe.issues[0].code == "unsafe_provider_path"
    assert not (tmp_path / "escape.jsonl").exists()


def test_unknown_license_stays_approval_gated_and_card_reference_is_pinned(
    tmp_path: Path,
) -> None:
    provider = FakeRemoteProvider(
        _snapshot(license=None, license_source=None, missing_metadata=("license",)),
        (_file(),),
    )
    service, _, _ = _service(tmp_path, provider=provider)
    candidate = _candidate(license=None)

    plan = _plan(service, candidate, _selection(candidate))

    assert f"unknown_license:{candidate.candidate_id}" in plan.required_approvals
    assert plan.snapshot.dataset_card_revision == REVISION
    assert REVISION in str(plan.snapshot.dataset_card_ref)

    generic_approval = _approval(candidate, _selection(candidate))
    blocked = service.acquire(plan, generic_approval)
    assert blocked.receipt.issues[0].code == "dataset_policy_approvals_missing"
    approved = DatasetAcquisitionApproval(
        generic_approval.selection_decision_id,
        generic_approval.approved_candidate_ids,
        generic_approval.approval_refs,
        approved_requirements=plan.required_approvals,
    )
    assert service.acquire(plan, approved).completed is True


def test_missing_or_wrong_approval_prevents_any_transfer(tmp_path: Path) -> None:
    service, _, transport = _service(tmp_path)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(service, candidate, selection)
    missing = DatasetAcquisitionApproval(selection.decision_id, (), ())

    result = service.acquire(plan, missing)

    assert result.receipt.completion_status == "failed"
    assert result.receipt.issues[0].code == "dataset_acquisition_approval_missing"
    assert transport.requests == []


def test_approved_streaming_acquisition_hashes_caches_artifacts_and_round_trips_receipt(
    tmp_path: Path,
) -> None:
    service, _, transport = _service(tmp_path, artifact_store=True)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(
        service,
        candidate,
        selection,
        limits=RemoteAcquisitionLimits(
            chunk_size=5, max_bytes=10_000, max_files=2, disk_reserve_bytes=0
        ),
    )

    result = service.acquire(plan, _approval(candidate, selection))

    assert result.completed is True
    evidence = result.receipt.acquired_files[0]
    assert evidence.local_content_hash == f"sha256:{hashlib.sha256(DATA).hexdigest()}"
    assert evidence.provider_hash_status == "verified"
    assert evidence.transport_checksum_status == "verified"
    assert evidence.cache_status == "miss_stored"
    assert evidence.artifact_ref == evidence.local_content_hash
    assert Path(evidence.cache_ref).read_bytes() == DATA
    assert len(transport.requests) == 1
    restored = AcquisitionReceipt.from_dict(result.receipt.to_dict())
    assert restored.to_dict() == result.receipt.to_dict()


def test_secret_reference_is_resolved_but_secret_is_never_persisted(
    tmp_path: Path,
) -> None:
    secret = "hf_super_secret_value"
    service, provider, transport = _service(tmp_path, secret_value=secret)
    reference = SecretReference("environment", "HF_DATASET_TOKEN")
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(service, candidate, selection, authentication_reference=reference)

    result = service.acquire(
        plan, _approval(candidate, selection), authentication_reference=reference
    )

    serialized = json.dumps(
        {"plan": plan.to_dict(), "receipt": result.receipt.to_dict()}
    )
    assert secret not in serialized
    assert plan.authentication_reference == reference.to_dict()
    assert secret in provider.tokens
    assert transport.requests[0]["headers"]["Authorization"] == f"Bearer {secret}"


def test_maximum_bytes_are_enforced_during_streaming_when_size_is_unknown(
    tmp_path: Path,
) -> None:
    unknown = _file(size_bytes=None, provider_hash=None, provider_hash_algorithm=None)
    provider = FakeRemoteProvider(_snapshot(), (unknown,))
    service, _, _ = _service(tmp_path, provider=provider)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(
        service,
        candidate,
        selection,
        limits=RemoteAcquisitionLimits(
            max_bytes=5, max_files=1, chunk_size=2, disk_reserve_bytes=0
        ),
    )

    result = service.acquire(plan, _approval(candidate, selection))

    assert result.completed is False
    assert result.receipt.issues[0].code == "maximum_byte_budget_exceeded"
    assert result.receipt.acquired_files == ()
    assert not list((tmp_path / "cache" / "partial").rglob("*.part"))


def test_interrupted_partial_is_not_complete_and_valid_range_resume_finishes(
    tmp_path: Path,
) -> None:
    failing = FakeTransport({URL: DATA}, fail_after=4)
    service, provider, _ = _service(tmp_path, transport=failing)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(
        service,
        candidate,
        selection,
        limits=RemoteAcquisitionLimits(
            chunk_size=2, max_bytes=10_000, max_files=1, disk_reserve_bytes=0
        ),
    )

    interrupted = service.acquire(plan, _approval(candidate, selection))
    assert interrupted.receipt.completion_status == "failed"
    assert interrupted.receipt.transfer_attempts[0].partial_preserved is True
    assert list((tmp_path / "cache" / "partial").rglob("*.part"))

    resumed_transport = FakeTransport({URL: DATA}, support_range=True)
    resumed = RemoteDatasetAcquisitionService(
        providers={provider.provider_id: provider},
        cache=service.cache,
        transport=resumed_transport,
    ).acquire(plan, _approval(candidate, selection))

    assert resumed.completed is True
    assert resumed_transport.requests[0]["headers"]["Range"] == "bytes=4-"
    assert resumed.receipt.partial_recovery_evidence[0]["action"] == "range_resume"
    assert Path(resumed.receipt.acquired_files[0].cache_ref).read_bytes() == DATA


def test_unsupported_range_restarts_safely_from_zero(tmp_path: Path) -> None:
    service, provider, _ = _service(tmp_path)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(service, candidate, selection)
    file = plan.files[0]
    state = service.cache.prepare_partial(
        provider_id=plan.snapshot.provider_id,
        dataset_id=plan.snapshot.dataset_id,
        resolved_revision=plan.snapshot.resolved_revision,
        file=file,
        byte_count=4,
        etag='"immutable-etag"',
    )
    state.path.write_bytes(DATA[:4])
    service.cache.prepare_partial(
        provider_id=plan.snapshot.provider_id,
        dataset_id=plan.snapshot.dataset_id,
        resolved_revision=plan.snapshot.resolved_revision,
        file=file,
        byte_count=4,
        etag='"immutable-etag"',
    )
    transport = FakeTransport({URL: DATA}, support_range=False)
    restarted = RemoteDatasetAcquisitionService(
        providers={provider.provider_id: provider},
        cache=service.cache,
        transport=transport,
    ).acquire(plan, _approval(candidate, selection))

    assert restarted.completed is True
    assert (
        restarted.receipt.partial_recovery_evidence[0]["action"] == "restart_from_zero"
    )
    assert Path(restarted.receipt.acquired_files[0].cache_ref).read_bytes() == DATA


def test_revision_change_invalidates_partial_before_network_transfer(
    tmp_path: Path,
) -> None:
    provider = FakeRemoteProvider(_snapshot(), (_file(),), current=False)
    service, _, transport = _service(tmp_path, provider=provider)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(service, candidate, selection)
    state = service.cache.prepare_partial(
        provider_id=plan.snapshot.provider_id,
        dataset_id=plan.snapshot.dataset_id,
        resolved_revision=plan.snapshot.resolved_revision,
        file=plan.files[0],
        byte_count=4,
        etag='"immutable-etag"',
    )
    state.path.write_bytes(DATA[:4])
    service.cache.prepare_partial(
        provider_id=plan.snapshot.provider_id,
        dataset_id=plan.snapshot.dataset_id,
        resolved_revision=plan.snapshot.resolved_revision,
        file=plan.files[0],
        byte_count=4,
        etag='"immutable-etag"',
    )

    result = service.acquire(plan, _approval(candidate, selection))

    assert result.receipt.issues[0].code == "revision_changed"
    assert transport.requests == []
    assert not state.path.exists()
    assert result.receipt.cleanup[0]["action"] == "partial_removed"


def test_changed_etag_during_resume_fails_safely_and_removes_partial(
    tmp_path: Path,
) -> None:
    service, _, _ = _service(tmp_path)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(service, candidate, selection)
    state = service.cache.prepare_partial(
        provider_id=plan.snapshot.provider_id,
        dataset_id=plan.snapshot.dataset_id,
        resolved_revision=plan.snapshot.resolved_revision,
        file=plan.files[0],
        byte_count=4,
        etag='"old-etag"',
    )
    state.path.write_bytes(DATA[:4])
    service.cache.prepare_partial(
        provider_id=plan.snapshot.provider_id,
        dataset_id=plan.snapshot.dataset_id,
        resolved_revision=plan.snapshot.resolved_revision,
        file=plan.files[0],
        byte_count=4,
        etag='"old-etag"',
    )

    result = service.acquire(plan, _approval(candidate, selection))

    assert result.receipt.issues[0].code == "revision_changed"
    assert result.receipt.acquired_files == ()
    assert not state.path.exists()


def test_provider_hash_mismatch_blocks_use_and_cleans_invalid_partial(
    tmp_path: Path,
) -> None:
    wrong = _file(provider_hash="0" * 64)
    provider = FakeRemoteProvider(_snapshot(), (wrong,))
    service, _, _ = _service(tmp_path, provider=provider)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(service, candidate, selection)

    result = service.acquire(plan, _approval(candidate, selection))

    assert result.receipt.issues[0].code == "checksum_mismatch"
    assert result.receipt.acquired_files == ()
    assert not list((tmp_path / "cache" / "partial").rglob("*.part"))


def test_cache_hit_is_verified_corruption_is_quarantined_and_offline_reuse_is_honest(
    tmp_path: Path,
) -> None:
    service, provider, _ = _service(tmp_path)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(service, candidate, selection)
    approval = _approval(candidate, selection)
    first = service.acquire(plan, approval)
    assert first.completed

    offline_transport = FakeTransport({URL: DATA})
    offline_service = RemoteDatasetAcquisitionService(
        providers={provider.provider_id: provider},
        cache=service.cache,
        transport=offline_transport,
    )
    offline = offline_service.acquire(plan, approval, offline=True)
    assert offline.completed is True
    assert offline.receipt.cache_status == "all_hits"
    assert offline_transport.requests == []

    Path(first.receipt.acquired_files[0].cache_ref).write_bytes(b"corrupt")
    redownload_transport = FakeTransport({URL: DATA})
    repaired = RemoteDatasetAcquisitionService(
        providers={provider.provider_id: provider},
        cache=service.cache,
        transport=redownload_transport,
    ).acquire(plan, approval)
    assert repaired.completed is True
    assert any(
        item["action"] == "corrupt_cache_quarantined"
        for item in repaired.receipt.cleanup
    )
    assert redownload_transport.requests


def test_offline_mode_fails_honestly_when_required_cache_entry_is_absent(
    tmp_path: Path,
) -> None:
    service, _, transport = _service(tmp_path)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(service, candidate, selection)

    result = service.acquire(plan, _approval(candidate, selection), offline=True)

    assert result.receipt.completion_status == "failed"
    assert result.receipt.issues[0].code == "offline_artifact_missing"
    assert transport.requests == []


def test_identical_offline_evidence_produces_same_receipt_id_despite_timestamps(
    tmp_path: Path,
) -> None:
    service, provider, _ = _service(tmp_path)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(service, candidate, selection)
    approval = _approval(candidate, selection)
    assert service.acquire(plan, approval).completed
    offline_service = RemoteDatasetAcquisitionService(
        providers={provider.provider_id: provider},
        cache=service.cache,
        transport=FakeTransport({URL: DATA}),
    )

    first = offline_service.acquire(plan, approval, offline=True).receipt
    second = offline_service.acquire(plan, approval, offline=True).receipt

    assert first.receipt_id == second.receipt_id
    assert first.deterministic_dict() == second.deterministic_dict()


def test_file_and_known_byte_limits_block_deterministically_at_planning(
    tmp_path: Path,
) -> None:
    second_url = URL.replace("train.jsonl", "other.jsonl")
    files = (_file(), _file(relative_path="data/other.jsonl", source_url=second_url))
    provider = FakeRemoteProvider(_snapshot(), files)
    service, _, _ = _service(
        tmp_path,
        provider=provider,
        transport=FakeTransport({URL: DATA, second_url: DATA}),
    )
    candidate = _candidate()
    selection = _selection(candidate)

    file_block = service.plan(
        candidate=candidate,
        selection=selection,
        limits=RemoteAcquisitionLimits(max_files=1, max_bytes=10_000),
    )
    byte_block = service.plan(
        candidate=candidate,
        selection=selection,
        limits=RemoteAcquisitionLimits(max_files=2, max_bytes=len(DATA)),
    )

    assert file_block.issues[0].code == "maximum_file_count_exceeded"
    assert byte_block.issues[0].code == "maximum_byte_budget_exceeded"
