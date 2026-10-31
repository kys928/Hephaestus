"""Configured public Hub registry; resolve real revision and original card bytes.

Registry membership is discovery permission, never data rights or admission.
"""
from dataclasses import dataclass
from hashlib import sha256
from urllib.parse import quote
from urllib.request import urlopen, Request
import time
from hephaestus.schemas.discovery_contract import DatasetCandidate
from hephaestus.policy.internal_research import put_record
from .huggingface import HuggingFaceApiClient, _license_name


@dataclass(frozen=True)
class ApprovedHubRegistryProvider:
    dataset_ids: tuple[str, ...]
    artifacts: object
    provider_id: str = "approved_hub_registry"

    def search(self, request):
        if not self.dataset_ids or len(self.dataset_ids) > 8:
            raise ValueError("explicit bounded public registry required")
        client = HuggingFaceApiClient(timeout_seconds=10)
        result = []
        for dataset_id in self.dataset_ids:
            info = client.dataset_info(dataset_id, "main", files_metadata=True)
            ref = put_record(self.artifacts, {"schema": "original-hub-metadata.v1", "dataset_id": dataset_id, "response": info})
            revision = str(info.get("sha", ""))
            card = info.get("cardData", {})
            license_name = _license_name(card.get("license"))
            paths = [r["rfilename"] for r in info.get("siblings", [])
                     if isinstance(r, dict) and str(r.get("rfilename", "")).endswith((".jsonl", ".parquet", ".json"))]
            result.append(DatasetCandidate("hub-registry-" + sha256((dataset_id + revision).encode()).hexdigest()[:24],
                self.provider_id, dataset_id, revision=revision, license=license_name,
                task_types=list(card.get("task_categories", [])), languages=list(card.get("language", [])),
                provenance={"metadata_ref": ref, "dataset_card_source": "original_publisher_hub"},
                evidence_refs=[ref], metadata={"acquisition_files": sorted(paths), "rights_accepted": False}))
        return tuple(result)


@dataclass
class EvidenceBoundSourceInspector:
    """Reviews configured sources. Unreviewed public rights fail and search continues.

    Prepared review refs may come from the governed original-data factory or a
    separate source-review service. Every byte and measurement is rechecked by
    admission, so this map cannot bless arbitrary pass flags.
    """
    artifacts: object
    review_refs: dict[str, str]

    def inspect(self, candidate, request, *, maximum_bytes, deadline):
        if time.monotonic() >= deadline:
            raise ValueError("source discovery deadline exhausted")
        ref = self.review_refs.get(candidate.candidate_id)
        if ref:
            from hephaestus.storage.verified import read_verified_json
            review = read_verified_json(self.artifacts, ref)
            total = 0
            for source in review["source_file_hashes"].values():
                with self.artifacts.open(source) as stream:
                    raw = stream.read(maximum_bytes - total + 1)
                total += len(raw)
                if total > maximum_bytes or "sha256:" + sha256(raw).hexdigest() != source:
                    raise ValueError("source bytes or discovery budget changed")
            return ref, total
        # Inspect actual publisher card, retaining original bytes in quarantine.
        # No source corpus is downloaded before rights/processing are reviewed.
        import re
        if candidate.provider_id == "approved_hub_registry" and re.fullmatch(r"[0-9a-f]{40}", candidate.revision or ""):
            url = "https://huggingface.co/datasets/" + quote(candidate.dataset_id, safe="/") + "/resolve/" + candidate.revision + "/README.md"
            with urlopen(Request(url, headers={"User-Agent": "hephaestus-internal-source-review/1"}),
                         timeout=max(0.1, min(10, deadline - time.monotonic()))) as response:
                raw = response.read(min(maximum_bytes, 256 * 1024) + 1)
            if len(raw) > min(maximum_bytes, 256 * 1024):
                raise ValueError("source card inspection budget exceeded")
            card = self.artifacts.put_bytes(raw, media_type="text/markdown").artifact_ref
            # This proves original bytes inspected. It does not interpret a
            # general repository license as permission for its data contents.
            evidence = put_record(self.artifacts, {"schema": "source-rights-triage.v1",
                "dataset_id": candidate.dataset_id, "revision": candidate.revision,
                "original_card_ref": card, "original_card_url": url,
                "declared_license": candidate.license, "data_rights_accepted": False,
                "disposition": "Tier B", "reason": "No accepted data-specific grant, full-source scan and prospective overlap review."})
            raise SourceReviewRequired(evidence, len(raw))
        raise PermissionError("no authentic full-source internal review available")


class SourceReviewRequired(PermissionError):
    def __init__(self, evidence_ref, consumed):
        super().__init__("source requires data-specific rights/full-corpus review: " + evidence_ref)
        self.evidence_ref, self.consumed = evidence_ref, consumed
