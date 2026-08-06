from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from hephaestus.providers.datasets import (
    DatasetProviderAcquisitionError,
    HuggingFaceDatasetProvider,
)
from hephaestus.schemas.discovery_contract import DatasetSearchRequest

REVISION = "b" * 40


@dataclass(slots=True)
class FakeHubClient:
    info_sha: str = REVISION
    calls: list[tuple[str, object]] = field(default_factory=list)

    def search_datasets(self, query: str, *, limit: int, token: str | None = None):
        self.calls.append(("search", {"query": query, "limit": limit, "token": token}))
        return [
            {
                "id": "org/dataset",
                "sha": REVISION,
                "tags": ["language:en", "task_categories:text-generation"],
                "cardData": {"license": "Apache-2.0"},
                "downloads": 10,
            }
        ]

    def dataset_info(
        self,
        dataset_id: str,
        revision: str,
        *,
        files_metadata: bool,
        token: str | None = None,
    ):
        self.calls.append(
            (
                "info",
                {
                    "dataset_id": dataset_id,
                    "revision": revision,
                    "files_metadata": files_metadata,
                    "token": token,
                },
            )
        )
        payload = {
            "id": dataset_id,
            "sha": self.info_sha,
            "private": False,
            "gated": False,
            "author": "Dataset Author",
            "lastModified": "2026-08-01T00:00:00Z",
            "cardData": {
                "license": "Apache-2.0",
                "citation": "Dataset citation",
                "authors": ["Author A"],
                "license_details": ["research and commercial use"],
            },
        }
        if files_metadata:
            payload["siblings"] = [
                {
                    "rfilename": "data/train.parquet",
                    "blobId": "c" * 40,
                    "size": 123,
                    "lfs": {
                        "sha256": "d" * 64,
                        "size": 123,
                        "oid": "sha256:" + "d" * 64,
                    },
                },
                {"rfilename": "README.md", "blobId": "e" * 40, "size": 40},
                {"rfilename": "dataset.py", "blobId": "f" * 40, "size": 20},
            ]
        return payload


def _request() -> DatasetSearchRequest:
    return DatasetSearchRequest(
        request_id="search-hf",
        diagnosis_report_id="diag-hf",
        problem_statement="English text generation",
        capability_targets=["text-generation"],
        provider_allowlist=["huggingface"],
    )


def test_huggingface_discovery_is_metadata_only_and_revision_aware() -> None:
    client = FakeHubClient()
    provider = HuggingFaceDatasetProvider(client=client)  # type: ignore[arg-type]

    candidate = provider.search(_request())[0]

    assert candidate.dataset_id == "org/dataset"
    assert candidate.revision == REVISION
    assert candidate.license == "apache-2.0"
    assert candidate.languages == ["en"]
    assert candidate.compatibility["remote_code_required"] is False
    assert candidate.compatibility["immutable_revision_available"] is True
    assert not any(call[0] == "info" for call in client.calls)


def test_huggingface_resolution_enumeration_and_card_evidence_are_commit_pinned() -> (
    None
):
    client = FakeHubClient()
    provider = HuggingFaceDatasetProvider(client=client)  # type: ignore[arg-type]

    snapshot = provider.resolve_revision("org/dataset", "main", token="runtime-only")
    files = provider.enumerate_files(snapshot, token="runtime-only")

    assert snapshot.requested_revision == "main"
    assert snapshot.resolved_revision == REVISION
    assert snapshot.dataset_card_revision == REVISION
    assert (
        snapshot.dataset_card_ref
        == f"https://huggingface.co/datasets/org/dataset/blob/{REVISION}/README.md"
    )
    assert snapshot.license == "apache-2.0"
    assert snapshot.license_source == "huggingface_dataset_card"
    assert snapshot.citation == "Dataset citation"
    assert files[0].relative_path == "README.md"
    data_file = next(item for item in files if item.relative_path.endswith(".parquet"))
    assert (
        data_file.source_url
        == f"https://huggingface.co/datasets/org/dataset/resolve/{REVISION}/data/train.parquet"
    )
    assert data_file.provider_hash == "d" * 64
    assert data_file.provider_hash_algorithm == "sha256"
    code_file = next(item for item in files if item.relative_path == "dataset.py")
    assert code_file.media_type == "application/x-python"
    assert all("runtime-only" not in str(snapshot.to_dict()) for _ in [0])


def test_huggingface_revision_verification_detects_floating_ref_change() -> None:
    client = FakeHubClient()
    provider = HuggingFaceDatasetProvider(client=client)  # type: ignore[arg-type]
    snapshot = provider.resolve_revision("org/dataset", "main")
    client.info_sha = "1" * 40

    assert provider.revision_is_current(snapshot) is False


def test_huggingface_refuses_non_immutable_resolution_and_revision_mismatch() -> None:
    client = FakeHubClient(info_sha="main")
    provider = HuggingFaceDatasetProvider(client=client)  # type: ignore[arg-type]
    with pytest.raises(
        DatasetProviderAcquisitionError, match="full commit SHA"
    ) as unresolved:
        provider.resolve_revision("org/dataset", "main")
    assert unresolved.value.code == "immutable_revision_unresolved"

    client.info_sha = REVISION
    snapshot = provider.resolve_revision("org/dataset", "main")
    client.info_sha = "2" * 40
    with pytest.raises(
        DatasetProviderAcquisitionError, match="did not match"
    ) as changed:
        provider.enumerate_files(snapshot)
    assert changed.value.code == "revision_changed"


def test_huggingface_network_is_disabled_by_default_without_optional_client() -> None:
    provider = HuggingFaceDatasetProvider(enable_network=False, client=None)

    with pytest.raises(DatasetProviderAcquisitionError) as unavailable:
        provider.search(_request())

    assert unavailable.value.code == "provider_unavailable"
