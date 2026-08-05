"""Model discovery providers that normalize registry metadata without approving it."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hephaestus.schemas.discovery_contract import ModelCandidate, ModelSearchRequest

_REQUIRED_METADATA = (
    "revision",
    "architecture_family",
    "parameter_count",
    "context_length",
    "tokenizer_ref",
    "license",
    "artifact_ref",
)


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted({str(item) for item in value if str(item).strip()})


def _string_set(value: object) -> set[str]:
    if isinstance(value, str):
        return {value} if value.strip() else set()
    return set(_as_str_list(value))


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class CatalogModelProvider:
    """Discover models from a local, reviewable catalog.

    Catalog entries are normalized into the shared contract. Missing facts stay
    missing; the provider never invents a revision, license, or compatibility.
    """

    entries: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    provider_id: str = "local_catalog"
    catalog_ref: str | None = None

    @classmethod
    def from_json(cls, path: str | Path, *, provider_id: str = "local_catalog") -> CatalogModelProvider:
        catalog_path = Path(path)
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        raw_entries = payload.get("models", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_entries, list) or not all(isinstance(item, dict) for item in raw_entries):
            raise ValueError("model catalog must be a list or an object with a models list")
        return cls(entries=tuple(raw_entries), provider_id=provider_id, catalog_ref=str(catalog_path))

    def search(self, request: ModelSearchRequest) -> Sequence[ModelCandidate]:
        if request.provider_allowlist and self.provider_id not in request.provider_allowlist:
            return []
        requested_ids = _string_set(request.metadata.get("model_ids"))
        requested_id = str(request.metadata.get("model_id", "")).strip()
        if requested_id:
            requested_ids.add(requested_id)

        candidates: list[ModelCandidate] = []
        for entry in self.entries:
            model_id = str(entry.get("model_id", "")).strip()
            if not model_id or (requested_ids and model_id not in requested_ids):
                continue
            candidate = self._normalize(entry)
            requested_revision = str(request.metadata.get("revision", "")).strip()
            if requested_revision and candidate.revision != requested_revision:
                candidate.risk_signals.append("requested_revision_unavailable")
                candidate.compatibility["requested_revision"] = False
            candidates.append(candidate)
        return tuple(sorted(candidates, key=lambda item: item.candidate_id))

    def _normalize(self, entry: Mapping[str, object]) -> ModelCandidate:
        model_id = str(entry["model_id"])
        revision = str(entry["revision"]).strip() if entry.get("revision") else None
        candidate_id = str(entry.get("candidate_id") or f"{self.provider_id}:{model_id}@{revision or 'unknown'}")
        missing = [name for name in _REQUIRED_METADATA if entry.get(name) in (None, "")]
        declared_missing = _as_str_list(entry.get("missing_metadata", []))
        score_components = {
            "metadata_completeness": round(1.0 - len(set(missing + declared_missing)) / len(_REQUIRED_METADATA), 6),
            "provider_evidence": 1.0 if self.catalog_ref or entry.get("evidence_refs") else 0.5,
        }
        raw_scores = entry.get("score_components")
        if isinstance(raw_scores, Mapping):
            for key, value in raw_scores.items():
                try:
                    score_components[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        compatibility = dict(entry.get("compatibility", {})) if isinstance(entry.get("compatibility"), Mapping) else {}
        runtime = dict(entry.get("runtime_requirements", {})) if isinstance(entry.get("runtime_requirements"), Mapping) else {}
        evidence_refs = _as_str_list(entry.get("evidence_refs", []))
        if self.catalog_ref:
            evidence_refs.append(self.catalog_ref)
        return ModelCandidate(
            candidate_id=candidate_id,
            provider_id=self.provider_id,
            model_id=model_id,
            revision=revision,
            architecture_family=str(entry["architecture_family"]) if entry.get("architecture_family") else None,
            parameter_count=_optional_int(entry.get("parameter_count")),
            context_length=_optional_int(entry.get("context_length")),
            tokenizer_ref=str(entry["tokenizer_ref"]) if entry.get("tokenizer_ref") else None,
            license=str(entry["license"]) if entry.get("license") else None,
            capabilities=_as_str_list(entry.get("capabilities", [])),
            runtime_requirements=runtime,
            compatibility=compatibility,
            risk_signals=_as_str_list(entry.get("risk_signals", [])),
            artifact_ref=str(entry["artifact_ref"]) if entry.get("artifact_ref") else None,
            evidence_refs=sorted(set(evidence_refs)),
            missing_metadata=sorted(set(missing + declared_missing)),
            score_components=score_components,
            metadata=dict(entry.get("metadata", {})) if isinstance(entry.get("metadata"), Mapping) else {},
        )


class FakeModelProvider(CatalogModelProvider):
    """Deterministic offline provider used by unit and smoke tests."""

    def __init__(self) -> None:
        super().__init__(
            provider_id="fake_models",
            catalog_ref="fixture://models/v1",
            entries=(
                {
                    "model_id": "tiny-char-lm",
                    "revision": "fixture-v1",
                    "architecture_family": "tiny_linear_lm",
                    "parameter_count": 2,
                    "context_length": 128,
                    "tokenizer_ref": "fixture://byte-tokenizer/v1",
                    "license": "internal-test-only",
                    "capabilities": ["causal_lm", "smoke_test"],
                    "runtime_requirements": {
                        "supported_backends": ["local_fixture", "fake"],
                        "memory_gb": 0.01,
                        "estimated_runtime_seconds": 2,
                    },
                    "compatibility": {"checkpoint_integrity": "content_hash", "smoke_test": True},
                    "artifact_ref": "fixture://tiny-char-lm/v1",
                },
                {
                    "model_id": "tiny-unknown-license",
                    "revision": "fixture-v1",
                    "architecture_family": "tiny_linear_lm",
                    "parameter_count": 2,
                    "context_length": 64,
                    "tokenizer_ref": "fixture://byte-tokenizer/v1",
                    "license": None,
                    "capabilities": ["causal_lm"],
                    "runtime_requirements": {"supported_backends": ["local_fixture"]},
                    "compatibility": {"smoke_test": True},
                    "artifact_ref": "fixture://tiny-unknown-license/v1",
                },
            ),
        )


@dataclass(slots=True)
class ExternalModelRegistryProvider(CatalogModelProvider):
    """Opt-in adapter around an injected external registry client.

    No network client or credential behavior is enabled by default. Integration
    code must explicitly construct this adapter with a fetch callable.
    """

    fetch_entries: Callable[[ModelSearchRequest], Iterable[Mapping[str, object]]] | None = None
    enabled: bool = False
    provider_id: str = "external_registry"

    def search(self, request: ModelSearchRequest) -> Sequence[ModelCandidate]:
        if not self.enabled:
            raise RuntimeError("external model registry adapter is opt-in and disabled")
        if self.fetch_entries is None:
            raise RuntimeError("external model registry adapter requires fetch_entries")
        self.entries = tuple(self.fetch_entries(request))
        return super().search(request)
