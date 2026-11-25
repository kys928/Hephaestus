"""Typed evaluation-only research objectives and independently reviewed sources."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping
import json
import re


_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")
_ALLOWED_CAPABILITIES = {"diagnosis_coverage_v1"}


class ResearchAdmissionError(ValueError):
    """A proposed objective/source could not be admitted by deterministic policy."""


@dataclass(frozen=True, slots=True)
class ResearchObjective:
    question: str
    capability: str = "diagnosis_coverage_v1"
    maximum_sources: int = 2
    seeds: tuple[int, ...] = (271828, 314159, 161803)

    @classmethod
    def from_text(cls, question: str) -> "ResearchObjective":
        """Natural-language intake for the first registered capability, not arbitrary agency.

        Unsupported goals are rejected; source selection and outcomes are never
        inferred from free-form text. A future model router may propose a
        capability, but the same admission requirements must apply.
        """
        if not isinstance(question, str) or not 16 <= len(question.strip()) <= 1000:
            raise ResearchAdmissionError("research objective text must be bounded")
        lowered = question.casefold()
        if not ("diagnos" in lowered and any(word in lowered for word in (
                "reliab", "cover", "evidence", "robust", "valid", "investigat", "experiment"))):
            raise ResearchAdmissionError("unsupported objective: no admitted diagnostic research capability")
        return cls(question.strip())

    def validate(self) -> None:
        if self.capability not in _ALLOWED_CAPABILITIES or self.maximum_sources != 2:
            raise ResearchAdmissionError("unsupported research capability or source ceiling")
        if (not isinstance(self.question, str) or not 16 <= len(self.question.strip()) <= 1000
                or not 1 <= len(self.seeds) <= 5
                or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds)):
            raise ResearchAdmissionError("invalid objective or frozen seeds")


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    source_id: str
    subject: str
    study_scope: str
    source_uri: str
    sha256_ref: str
    expected_rows: int
    expected_bytes: int
    role: str
    capability: str

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "SourceDescriptor":
        item = cls(**dict(row))
        if (not _SHA.fullmatch(item.sha256_ref) or item.role not in {"original", "candidate"}
                or item.capability not in _ALLOWED_CAPABILITIES
                or not item.study_scope or not item.subject or not item.source_id
                or not item.source_uri.startswith("runpod-s3://")
                or type(item.expected_rows) is not int or not 0 < item.expected_rows <= 1_000_000
                or type(item.expected_bytes) is not int or not 0 < item.expected_bytes <= 256_000_000):
            raise ResearchAdmissionError("invalid or inadmissible source descriptor")
        return item


@dataclass(frozen=True, slots=True)
class SourceCatalog:
    """An externally pinned, *reviewed* catalog; catalog fields aren't self-proof."""

    sources: tuple[SourceDescriptor, ...]
    catalog_sha256: str

    @classmethod
    def load(cls, path: Path, *, independently_pinned_sha256: str) -> "SourceCatalog":
        data = Path(path).read_bytes()
        if (not _SHA.fullmatch(independently_pinned_sha256)
                or "sha256:" + sha256(data).hexdigest() != independently_pinned_sha256):
            raise ResearchAdmissionError("source catalog does not match independently pinned bytes")
        payload = json.loads(data)
        if not isinstance(payload, dict) or payload.get("schema") != "research-source-catalog.v1":
            raise ResearchAdmissionError("unsupported source catalog schema")
        records = payload.get("sources")
        if not isinstance(records, list) or not 2 <= len(records) <= 32:
            raise ResearchAdmissionError("invalid source catalog length")
        sources = tuple(SourceDescriptor.from_dict(item) for item in records)
        if len({item.source_id for item in sources}) != len(sources):
            raise ResearchAdmissionError("duplicate source descriptor identity")
        return cls(sources, independently_pinned_sha256)

    def select(self, objective: ResearchObjective) -> tuple[SourceDescriptor, SourceDescriptor]:
        objective.validate()
        # Source discovery is from the admitted catalog, not arbitrary S3 listing
        # or a model-provided object key. Never combine unrelated research lines.
        scoped = [source for source in self.sources if source.capability == objective.capability]
        families: dict[str, dict[str, SourceDescriptor]] = {}
        for source in scoped:
            family = families.setdefault(source.study_scope, {})
            if source.role in family:
                raise ResearchAdmissionError("ambiguous duplicate research source role")
            family[source.role] = source
        eligible = [(key, value) for key, value in families.items()
                    if set(value) == {"original", "candidate"}]
        if len(eligible) != 1:
            raise ResearchAdmissionError("expected one uniquely matched reviewed source family")
        _, match = eligible[0]
        return match["original"], match["candidate"]


@dataclass(frozen=True, slots=True)
class ReviewedSourceJob:
    intake_ref: str
    run_id: str
    lineage_id: str
    catalog_ref: str
    eval_pack_ref: str
    schema: str = "hephaestus.reviewed-source-job.v1"

    def __post_init__(self) -> None:
        if (self.schema != "hephaestus.reviewed-source-job.v1"
                or not all(isinstance(value, str) and _SHA.fullmatch(value)
                           for value in (self.intake_ref, self.catalog_ref, self.eval_pack_ref))
                or not all(isinstance(value, str) and value.strip() and len(value) <= 128
                           for value in (self.run_id, self.lineage_id))):
            raise ResearchAdmissionError("invalid reviewed source job scope")
