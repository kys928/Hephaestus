from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import quote
from urllib.request import Request, urlopen

from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSearchRequest
from hephaestus.utils.hashing import hash_json


@dataclass(slots=True)
class HuggingFaceDatasetProvider:
    """Optional metadata-only Hugging Face Hub adapter.

    Network access is disabled by default. The adapter only reads Hub metadata;
    it never imports a dataset script or enables remote code execution.
    """

    provider_id: str = "huggingface"
    enable_network: bool = False
    max_results: int = 10
    timeout_seconds: float = 10.0
    endpoint: str = "https://huggingface.co/api/datasets"

    def search(self, request: DatasetSearchRequest) -> tuple[DatasetCandidate, ...]:
        if not self.enable_network:
            raise RuntimeError("Hugging Face provider network access is disabled")
        query = " ".join([request.problem_statement, *request.capability_targets]).strip()
        url = f"{self.endpoint}?search={quote(query)}&limit={max(1, min(self.max_results, 100))}&full=true"
        http_request = Request(url, headers={"Accept": "application/json", "User-Agent": "hephaestus-data-factory/1"})
        with urlopen(http_request, timeout=self.timeout_seconds) as response:  # noqa: S310 - explicit opt-in endpoint
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError("Hugging Face metadata response was not a list")

        candidates: list[DatasetCandidate] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            dataset_id = str(item.get("id") or "").strip()
            if not dataset_id:
                continue
            card = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
            tags = [str(tag) for tag in item.get("tags", []) if isinstance(tag, str)]
            languages = [tag.removeprefix("language:") for tag in tags if tag.startswith("language:")]
            task_types = [tag.removeprefix("task_categories:") for tag in tags if tag.startswith("task_categories:")]
            revision = str(item.get("sha") or "").strip() or None
            license_name = str(card.get("license") or "").strip() or None
            seed = {"provider_id": self.provider_id, "dataset_id": dataset_id, "revision": revision}
            candidates.append(
                DatasetCandidate(
                    candidate_id=f"dataset-{hash_json(seed)[:16]}",
                    provider_id=self.provider_id,
                    dataset_id=dataset_id,
                    revision=revision,
                    task_types=task_types,
                    languages=languages,
                    license=license_name,
                    provenance={"kind": "hub_metadata", "hub_id": dataset_id, "sha": revision},
                    trust_level="external_metadata",
                    compatibility={"remote_code_required": False, "metadata_only": True},
                    risk_signals=["requires_explicit_acquisition_enablement"],
                    evidence_refs=[f"https://huggingface.co/datasets/{dataset_id}"],
                    metadata={"downloads": item.get("downloads"), "likes": item.get("likes")},
                )
            )
        return tuple(candidates)
