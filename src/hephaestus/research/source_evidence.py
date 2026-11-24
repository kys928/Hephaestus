"""Verified inputs for a bounded evaluation-only diagnosis study."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Protocol
import json

from hephaestus.diagnosis.probes import analyze_dataset_task_coverage
from hephaestus.schemas.research_objective import ResearchAdmissionError, SourceDescriptor
from hephaestus.state.execution_ownership import execution_owner
from hephaestus.storage.base import ArtifactStore, StateRepository
from hephaestus.storage.verified import canonical_bytes

_SOURCE_COLLECTION = "research_source_cases"


class SourceReader(Protocol):
    def read(self, uri: str, *, maximum_bytes: int) -> bytes: ...


@dataclass(slots=True)
class ReadOnlyS3SourceReader:
    """Exact-key GET only. Never lists buckets, mutates remote state, or logs bytes."""

    client: object
    allowed_bucket: str

    def read(self, uri: str, *, maximum_bytes: int) -> bytes:
        if not uri.startswith("runpod-s3://"):
            raise ResearchAdmissionError("unsupported evidence source scheme")
        location = uri[len("runpod-s3://"):]
        bucket, sep, key = location.partition("/")
        if (not sep or bucket != self.allowed_bucket or not key.startswith("hephaestus/scientific/v1/")
                or ".." in key or "\\" in key):
            raise ResearchAdmissionError("evidence object outside reviewed S3 scope")
        response = self.client.get_object(Bucket=bucket, Key=key)
        stream = response["Body"]
        chunks = []
        count = 0
        try:
            while True:
                block = stream.read(min(1024 * 1024, maximum_bytes + 1 - count))
                if not block:
                    break
                count += len(block)
                if count > maximum_bytes:
                    raise ResearchAdmissionError("source exceeds its pinned byte ceiling")
                chunks.append(block)
        finally:
            stream.close()
        return b"".join(chunks)


@dataclass(slots=True)
class VerifiedSourceAcquirer:
    reader: SourceReader
    artifacts: ArtifactStore
    repository: StateRepository

    def acquire(self, source: SourceDescriptor, eval_pack: Mapping[str, object]) -> dict[str, object]:
        # This read-only operation is replay-safe. Read bytes, hash *those bytes*,
        # then parse; never trust ContentLength or JSONL metadata.
        raw = self.reader.read(source.source_uri, maximum_bytes=source.expected_bytes)
        if (len(raw) != source.expected_bytes
                or "sha256:" + sha256(raw).hexdigest() != source.sha256_ref):
            raise ResearchAdmissionError("source bytes differ from independently reviewed identity")

        def records():
            for line in raw.splitlines():
                if line.strip():
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ResearchAdmissionError("JSONL row is not an object")
                    yield item

        evidence, measured = analyze_dataset_task_coverage(
            records(), eval_pack, source_ref=source.sha256_ref)
        if measured["rows_scanned"] != source.expected_rows:
            raise ResearchAdmissionError("source row count differs from reviewed identity")
        if measured["coverage_probe_version"] != 2:
            raise ResearchAdmissionError("outdated coverage observation would hide wrapper supervision")
        record = {
            "schema": "research-source-observation.v1",
            "source_id": source.source_id, "subject": source.subject,
            "study_scope": source.study_scope, "role": source.role,
            "source_ref": source.sha256_ref, "catalog_capability": source.capability,
            "counts": {
                name: measured[name] for name in (
                    "rows_scanned", "prompt_target_rows", "explicit_prompt_target_rows",
                    "embedded_marker_pair_rows", "embedded_valid_prompt_target_rows",
                    "embedded_empty_or_malformed_rows", "structured_target_rows",
                )
            },
            "supervised_instruction_coverage_present": measured["supervised_instruction_coverage_present"],
            "supervised_structure_coverage_present": measured["supervised_structure_coverage_present"],
            "data_coverage_gap": evidence["data_coverage_gap"],
            "measurement_version": 2,
            "original_bytes_reverified": True,
            "scientific_gold_label": None,
            "certification": "not_independently_adjudicated",
        }
        anchor = self.artifacts.put_bytes(canonical_bytes(record), media_type="application/json")
        record["observation_ref"] = anchor.artifact_ref
        with execution_owner(self.repository, "research-source-observation", anchor.artifact_ref):
            prior = self.repository.get_latest(_SOURCE_COLLECTION, "observation_ref", anchor.artifact_ref)
            if prior is None:
                self.repository.append(_SOURCE_COLLECTION, record)
            elif prior != record:
                raise ResearchAdmissionError("persisted source observation changed under same digest")
        return record
