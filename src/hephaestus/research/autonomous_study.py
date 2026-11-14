"""Bounded, training-free research study: objective -> authentic evidence -> role inference -> report.

This is an opt-in research job, not a fake 45-step training campaign. Original
trainable bytes are inspected read-only; protected eval data and weights never
enter this subsystem. An admitted study never authorizes a repair/promotion.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Protocol
import json
import re

from hephaestus.diagnosis.probes import analyze_dataset_task_coverage
from hephaestus.storage.base import ArtifactStore, StateRepository
from hephaestus.storage.verified import canonical_bytes, read_verified_json


_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")
_SOURCE_COLLECTION = "research_source_cases"
_STUDY_COLLECTION = "research_evaluation_studies"
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
        prior = self.repository.get_latest(_SOURCE_COLLECTION, "observation_ref", anchor.artifact_ref)
        if prior is None:
            self.repository.append(_SOURCE_COLLECTION, record)
        elif prior != record:
            raise ResearchAdmissionError("persisted source observation changed under same digest")
        return record


class FrozenRoleObserver(Protocol):
    def observe(self, *, objective: ResearchObjective,
                observation: Mapping[str, object], seed: int, run_id: str,
                lineage_id: str) -> Mapping[str, object]: ...


@dataclass(slots=True)
class CertifiedDiagnosisObserver:
    """Real frozen role inference, with exact original inference receipts."""

    reasoner: object
    stage_name: str = "smoke_test"

    def observe(self, *, objective: ResearchObjective, observation: Mapping[str, object],
                seed: int, run_id: str, lineage_id: str) -> Mapping[str, object]:
        from hephaestus.llm.role_contract import RoleEvidence, RoleTask
        from hephaestus.control.deterministic_role_boundary import (
            DeterministicRoleContext, EvidenceRecord, EvidenceRegistry,
        )
        ref = str(observation["observation_ref"])
        summary = {
            "source_subject": observation["subject"],
            "source_ref": observation["source_ref"],
            "observations": observation["counts"],
            "instruction_format_present": observation["supervised_instruction_coverage_present"],
            "structured_json_present": observation["supervised_structure_coverage_present"],
            "coverage_gap": observation["data_coverage_gap"],
            "model_evaluation_integrity": "not established by this source",
        }
        task = RoleTask(
            role="diagnosis", run_id=run_id, lineage_id=lineage_id,
            stage_name=self.stage_name,
            situation=objective.question + ". Do not assert an original Granite failure or a repaired model from a dataset-format measurement.",
            evidence=(RoleEvidence(ref, canonical_bytes(summary).decode("utf-8")),),
        )
        registry = EvidenceRegistry({ref: EvidenceRecord(
            ref=ref, kind="research_source_observation", run_id=run_id,
            lineage_id=lineage_id, sha256=ref, observed_sha256=ref)})
        result = self.reasoner.propose(task, registry, context=DeterministicRoleContext(
            run_id=run_id, lineage_id=lineage_id, stage_name=self.stage_name), seed=seed)
        return {
            "decision": result.answer.decision,
            "action": result.answer.action,
            "primary_variable": result.answer.primary_variable,
            "confidence": result.answer.confidence,
            "cited_refs": list(result.answer.evidence_refs),
            "receipt_ref": result.inference_receipt_ref,
            "model_id": result.model_id,
            "adapter_sha256": result.adapter_sha256,
        }


@dataclass(slots=True)
class EvaluationOnlyResearchJob:
    """One admitted read-only study, not another scheduler or a training stage."""

    catalog: SourceCatalog
    acquirer: VerifiedSourceAcquirer
    observer: FrozenRoleObserver
    repository: StateRepository
    artifacts: ArtifactStore
    eval_pack: Mapping[str, object]

    def run(self, question: str, *, run_id: str, lineage_id: str) -> dict[str, object]:
        objective = ResearchObjective.from_text(question)
        objective.validate()
        if not run_id or not lineage_id:
            raise ResearchAdmissionError("scientific run and lineage identity required")
        sources = self.catalog.select(objective)
        study_id = "research-" + sha256(canonical_bytes({
            "question": objective.question, "run_id": run_id,
            "lineage_id": lineage_id, "catalog": self.catalog.catalog_sha256,
            "seeds": objective.seeds,
        })).hexdigest()[:24]
        prior = self.repository.get_latest(_STUDY_COLLECTION, "study_id", study_id)
        if prior is not None:
            ref = str(prior.get("report_ref") or "")
            report = read_verified_json(self.artifacts, ref)
            if report.get("study_id") != study_id:
                raise ResearchAdmissionError("previous study identity mismatch")
            return report

        # Missing evidence is an explicit block; do not authorize training or
        # edit frozen evaluation inputs to improve a model's apparent score.
        observations = [self.acquirer.acquire(item, self.eval_pack) for item in sources]
        if observations[0]["study_scope"] != observations[1]["study_scope"]:
            raise ResearchAdmissionError("unmatched scientific source families")
        refs = [str(record["observation_ref"]) for record in observations]
        results = []
        for observation in observations:
            for seed in objective.seeds:
                output = dict(self.observer.observe(objective=objective, observation=observation,
                    seed=seed, run_id=run_id, lineage_id=lineage_id))
                if (not isinstance(output.get("decision"), str)
                        or not output["decision"] or not _SHA.fullmatch(str(output.get("receipt_ref") or ""))
                        or output.get("cited_refs") != [observation["observation_ref"]]
                        or not self.artifacts.verify(output["receipt_ref"])):
                    raise ResearchAdmissionError("role inference lacks verified exact-source receipt")
                if type(self.observer) is CertifiedDiagnosisObserver:
                    # Independently re-open the exact original role-generation
                    # receipt; an arbitrary sha256 object is not model proof.
                    from hephaestus.llm.inference_receipts import verify_inference_receipt
                    receipt=verify_inference_receipt(
                        self.artifacts,output["receipt_ref"],self.observer.reasoner.contract)
                    if (receipt.get("schema")!="role-inference.v1"
                            or receipt.get("status")!="accepted"
                            or receipt.get("role")!="diagnosis"
                            or receipt.get("run_id")!=run_id
                            or receipt.get("lineage_id")!=lineage_id
                            or receipt.get("seed")!=seed
                            or receipt.get("evidence_refs")!=[observation["observation_ref"]]
                            or receipt.get("model_id")!=output.get("model_id")
                            or receipt.get("adapter_sha256")!=output.get("adapter_sha256")
                            or receipt["answer"]["decision"]!=output["decision"]):
                        raise ResearchAdmissionError("frozen role receipt does not match admitted trial")
                results.append({
                    "source_id": observation["source_id"], "seed": seed,
                    "decision": output["decision"], "action": output.get("action"),
                    "primary_variable": output.get("primary_variable"),
                    "confidence": output.get("confidence"),
                    "receipt_ref": output["receipt_ref"],
                    "model_id": output.get("model_id"),
                    "adapter_sha256": output.get("adapter_sha256"),
                })
        counts = {
            src["source_id"]: dict(sorted(Counter(
                row["decision"] for row in results if row["source_id"] == src["source_id"]
            ).items())) for src in observations
        }
        report = {
            "schema": "research-evaluation-only-report.v1", "study_id": study_id,
            "objective": objective.question, "capability": objective.capability,
            "run_id": run_id, "lineage_id": lineage_id, "catalog_ref": self.catalog.catalog_sha256,
            "source_observation_refs": refs,
            "verified_case_count": len(observations),
            "seeds": list(objective.seeds), "role_invocations": len(results),
            "role_results": results, "decision_distributions": counts,
            "observed_source_difference": {
                "instruction_pairs_delta": observations[1]["counts"]["prompt_target_rows"] -
                    observations[0]["counts"]["prompt_target_rows"],
                "structured_targets_delta": observations[1]["counts"]["structured_target_rows"] -
                    observations[0]["counts"]["structured_target_rows"],
                "original_gap": observations[0]["data_coverage_gap"],
                "candidate_gap": observations[1]["data_coverage_gap"],
            },
            "role_inference_scope": (
                "genuine_certified_frozen_role_receipts" if type(self.observer) is CertifiedDiagnosisObserver
                else "injected_observer_not_frozen_model_verified"),
            "training_performed": False, "model_modified": False,
            "conclusion": "retrospective_development_observation_not_gold_label_validation",
            "independent_scientific_certification": False,
            "recommended_next_action": "assess_frozen_role_predictions_against_separately_admitted_gold",
            "status": "completed",
        }
        artifact = self.artifacts.put_bytes(canonical_bytes(report), media_type="application/json")
        self.repository.append(_STUDY_COLLECTION, {
            "study_id": study_id, "report_ref": artifact.artifact_ref, "status": "completed"})
        return report
