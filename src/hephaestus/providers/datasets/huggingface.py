from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSearchRequest
from hephaestus.utils.hashing import hash_json

from .acquisition import (
    DatasetProviderAcquisitionError,
    ProviderDatasetFile,
    ProviderDatasetSnapshot,
)

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")


@dataclass(slots=True)
class HuggingFaceApiClient:
    """Small optional-dependency-free client for read-only Hub metadata."""

    endpoint: str = "https://huggingface.co"
    timeout_seconds: float = 10.0
    user_agent: str = "hephaestus-production-data-acquisition/1"

    def search_datasets(
        self, query: str, *, limit: int, token: str | None = None
    ) -> list[object]:
        payload = self._get_json(
            f"/api/datasets?{urlencode({'search': query, 'limit': limit, 'full': 'true'})}",
            token=token,
        )
        if not isinstance(payload, list):
            raise DatasetProviderAcquisitionError(
                "malformed_provider_metadata",
                "Hugging Face dataset search response was not a list",
                category="internal_contract_violation",
            )
        return payload

    def dataset_info(
        self,
        dataset_id: str,
        revision: str,
        *,
        files_metadata: bool,
        token: str | None = None,
    ) -> dict[str, object]:
        dataset_path = quote(dataset_id, safe="/")
        revision_path = quote(revision, safe="")
        suffix = "?blobs=true" if files_metadata else ""
        payload = self._get_json(
            f"/api/datasets/{dataset_path}/revision/{revision_path}{suffix}",
            token=token,
        )
        if not isinstance(payload, dict):
            raise DatasetProviderAcquisitionError(
                "malformed_provider_metadata",
                "Hugging Face dataset info response was not an object",
                category="internal_contract_violation",
            )
        return {str(key): value for key, value in payload.items()}

    def _get_json(self, path: str, *, token: str | None) -> object:
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urlopen(
                Request(f"{self.endpoint.rstrip('/')}{path}", headers=headers),
                timeout=self.timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error = _provider_http_error(exc.code)
            raise error from exc
        except TimeoutError as exc:
            raise DatasetProviderAcquisitionError(
                "provider_timeout",
                "Hugging Face metadata request timed out",
                retryable=True,
            ) from exc
        except URLError as exc:
            raise DatasetProviderAcquisitionError(
                "provider_unavailable",
                "Hugging Face metadata endpoint is unavailable",
                retryable=True,
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetProviderAcquisitionError(
                "malformed_provider_metadata",
                "Hugging Face metadata response was not valid JSON",
                category="internal_contract_violation",
            ) from exc


@dataclass(slots=True)
class HuggingFaceDatasetProvider:
    """Governed Hugging Face discovery and immutable acquisition metadata adapter.

    Network use is disabled by default. This adapter reads metadata and resolves
    immutable file URLs only; it never imports repository code, dataset scripts,
    builders, dynamic modules, or ``trust_remote_code`` paths.
    """

    provider_id: str = "huggingface"
    enable_network: bool = False
    max_results: int = 10
    timeout_seconds: float = 10.0
    endpoint: str = "https://huggingface.co"
    client: HuggingFaceApiClient | None = field(default=None, repr=False)

    def _client(self) -> HuggingFaceApiClient:
        if not self.enable_network and self.client is None:
            raise DatasetProviderAcquisitionError(
                "provider_unavailable",
                "Hugging Face provider network access is disabled",
                retryable=False,
            )
        return self.client or HuggingFaceApiClient(self.endpoint, self.timeout_seconds)

    def search(self, request: DatasetSearchRequest) -> tuple[DatasetCandidate, ...]:
        query = " ".join(
            [request.problem_statement, *request.capability_targets]
        ).strip()
        payload = self._client().search_datasets(
            query,
            limit=max(1, min(self.max_results, 100)),
        )
        candidates: list[DatasetCandidate] = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            item = {str(key): value for key, value in raw.items()}
            dataset_id = str(item.get("id") or "").strip()
            if not dataset_id:
                continue
            card = (
                item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
            )
            tags = [str(tag) for tag in item.get("tags", []) if isinstance(tag, str)]
            languages = [
                tag.removeprefix("language:")
                for tag in tags
                if tag.startswith("language:")
            ]
            task_types = [
                tag.removeprefix("task_categories:")
                for tag in tags
                if tag.startswith("task_categories:")
            ]
            revision = str(item.get("sha") or "").strip() or None
            license_name = _license_name(card.get("license"))
            remote_code = _remote_code_declared(card, item)
            seed = {
                "provider_id": self.provider_id,
                "dataset_id": dataset_id,
                "revision": revision,
            }
            candidates.append(
                DatasetCandidate(
                    candidate_id=f"dataset-{hash_json(seed)[:16]}",
                    provider_id=self.provider_id,
                    dataset_id=dataset_id,
                    revision=revision,
                    task_types=task_types,
                    languages=languages,
                    license=license_name,
                    provenance={
                        "kind": "hub_metadata",
                        "hub_id": dataset_id,
                        "sha": revision,
                    },
                    trust_level="external_metadata",
                    compatibility={
                        "remote_code_required": remote_code,
                        "metadata_only": False,
                        "immutable_revision_available": bool(
                            revision and _IMMUTABLE_REVISION.fullmatch(revision)
                        ),
                    },
                    risk_signals=(
                        ["remote_dataset_code_required"]
                        if remote_code
                        else ["requires_explicit_acquisition_enablement"]
                    ),
                    evidence_refs=[
                        f"{self.endpoint.rstrip('/')}/datasets/{dataset_id}"
                    ],
                    missing_metadata=[
                        name
                        for name, value in (
                            ("revision", revision),
                            ("license", license_name),
                        )
                        if not value
                    ],
                    metadata={
                        "downloads": item.get("downloads"),
                        "likes": item.get("likes"),
                        "gated": item.get("gated", False),
                        "private": bool(item.get("private", False)),
                    },
                )
            )
        return tuple(candidates)

    def resolve_revision(
        self,
        dataset_id: str,
        requested_revision: str,
        *,
        token: str | None = None,
    ) -> ProviderDatasetSnapshot:
        requested = requested_revision.strip() or "main"
        info = self._client().dataset_info(
            dataset_id, requested, files_metadata=False, token=token
        )
        resolved = str(info.get("sha") or "").strip().lower()
        if not _IMMUTABLE_REVISION.fullmatch(resolved):
            raise DatasetProviderAcquisitionError(
                "immutable_revision_unresolved",
                "Hugging Face did not resolve the requested dataset revision to a full commit SHA",
                category="missing_evidence",
                metadata={"dataset_id": dataset_id, "requested_revision": requested},
            )
        card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
        license_name = _license_name(card.get("license"))
        missing = []
        if not license_name:
            missing.append("license")
        if not card:
            missing.append("dataset_card_metadata")
        terms = _string_tuple(
            card.get("license_details")
            or card.get("terms")
            or card.get("usage_restrictions")
        )
        authors = _string_tuple(card.get("authors") or info.get("author"))
        dataset_url = f"{self.endpoint.rstrip('/')}/datasets/{dataset_id}"
        return ProviderDatasetSnapshot(
            provider_id=self.provider_id,
            dataset_id=dataset_id,
            requested_revision=requested,
            resolved_revision=resolved,
            dataset_card_ref=f"{dataset_url}/blob/{resolved}/README.md",
            dataset_card_revision=resolved,
            license=license_name,
            license_source="huggingface_dataset_card" if license_name else None,
            terms=terms,
            citation=str(card.get("citation") or "").strip() or None,
            authors=authors,
            gated=bool(info.get("gated", False)),
            private=bool(info.get("private", False)),
            remote_code_required=_remote_code_declared(card, info),
            provenance_confidence="provider_commit_metadata",
            missing_metadata=tuple(sorted(missing)),
            metadata={
                "provider_object": dataset_id,
                "last_modified": info.get("lastModified"),
                "card_data_present": bool(card),
            },
        )

    def enumerate_files(
        self,
        snapshot: ProviderDatasetSnapshot,
        *,
        token: str | None = None,
    ) -> tuple[ProviderDatasetFile, ...]:
        info = self._client().dataset_info(
            snapshot.dataset_id,
            snapshot.resolved_revision,
            files_metadata=True,
            token=token,
        )
        returned_revision = str(info.get("sha") or "").strip().lower()
        if returned_revision != snapshot.resolved_revision:
            raise DatasetProviderAcquisitionError(
                "revision_changed",
                "provider file enumeration did not match the resolved immutable revision",
                category="artifact_integrity",
            )
        siblings = info.get("siblings")
        if not isinstance(siblings, list):
            raise DatasetProviderAcquisitionError(
                "malformed_provider_metadata",
                "Hugging Face dataset file metadata is missing",
                category="internal_contract_violation",
            )
        files: list[ProviderDatasetFile] = []
        for raw in siblings:
            if not isinstance(raw, dict):
                continue
            relative_path = str(raw.get("rfilename") or raw.get("path") or "").strip()
            if not relative_path:
                continue
            lfs = raw.get("lfs") if isinstance(raw.get("lfs"), dict) else {}
            provider_hash, algorithm = _provider_hash(raw, lfs)
            size = _optional_int(lfs.get("size") if lfs else raw.get("size"))
            source_url = (
                f"{self.endpoint.rstrip('/')}/datasets/{snapshot.dataset_id}/resolve/"
                f"{snapshot.resolved_revision}/{quote(relative_path, safe='/')}"
            )
            suffix = relative_path.casefold()
            media_type = "application/x-python" if suffix.endswith(".py") else None
            files.append(
                ProviderDatasetFile(
                    relative_path=relative_path,
                    source_url=source_url,
                    size_bytes=size,
                    provider_hash=provider_hash,
                    provider_hash_algorithm=algorithm,
                    etag=str(raw.get("etag") or "").strip() or None,
                    object_id=str(raw.get("blobId") or lfs.get("oid") or "").strip()
                    or None,
                    media_type=media_type,
                )
            )
        return tuple(sorted(files, key=lambda item: item.relative_path))

    def revision_is_current(
        self,
        snapshot: ProviderDatasetSnapshot,
        *,
        token: str | None = None,
    ) -> bool:
        info = self._client().dataset_info(
            snapshot.dataset_id,
            snapshot.requested_revision,
            files_metadata=False,
            token=token,
        )
        return str(info.get("sha") or "").strip().lower() == snapshot.resolved_revision


def _provider_http_error(status: int) -> DatasetProviderAcquisitionError:
    if status == 401:
        return DatasetProviderAcquisitionError(
            "authentication_failure", "Hugging Face authentication failed"
        )
    if status == 403:
        return DatasetProviderAcquisitionError(
            "gated_access_denied", "Hugging Face dataset access was denied"
        )
    if status == 404:
        return DatasetProviderAcquisitionError(
            "dataset_or_revision_not_found",
            "Hugging Face dataset or revision was not found",
        )
    if status == 429:
        return DatasetProviderAcquisitionError(
            "rate_limited", "Hugging Face rate limit was reached", retryable=True
        )
    if status >= 500:
        return DatasetProviderAcquisitionError(
            "provider_unavailable",
            "Hugging Face provider is unavailable",
            retryable=True,
        )
    return DatasetProviderAcquisitionError(
        "provider_error", f"Hugging Face metadata request failed with HTTP {status}"
    )


def _license_name(value: object) -> str | None:
    if isinstance(value, list):
        names = sorted(
            {str(item).strip().casefold() for item in value if str(item).strip()}
        )
        return ",".join(names) or None
    text = str(value or "").strip().casefold()
    return text or None


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted({str(item).strip() for item in value if str(item).strip()}))
    text = str(value or "").strip()
    return (text,) if text else ()


def _remote_code_declared(
    card: dict[object, object], info: dict[object, object]
) -> bool:
    keys = (
        "trust_remote_code",
        "requires_custom_code",
        "remote_code_required",
        "requires_remote_code",
    )
    return any(bool(card.get(key) or info.get(key)) for key in keys)


def _provider_hash(
    raw: dict[object, object], lfs: dict[object, object]
) -> tuple[str | None, str | None]:
    if lfs:
        value = (
            str(lfs.get("sha256") or lfs.get("oid") or "")
            .removeprefix("sha256:")
            .strip()
            .lower()
        )
        if re.fullmatch(r"[0-9a-f]{64}", value):
            return value, "sha256"
    blob_id = str(raw.get("blobId") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", blob_id):
        return blob_id, "git-blob-sha1"
    return None, None


def _optional_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


__all__ = ["HuggingFaceApiClient", "HuggingFaceDatasetProvider"]
