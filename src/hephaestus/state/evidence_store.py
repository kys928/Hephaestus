"""Durable evidence references, without caller-authored scientific summaries."""
from dataclasses import dataclass
import hashlib
from hephaestus.storage.verified import canonical_bytes
from hephaestus.schemas.scientific_evidence import ScientificEvidenceBinding
from hephaestus.storage.base import ArtifactStore, StateRepository
from hephaestus.storage.verified import read_verified_json

EVIDENCE_BINDINGS = "scientific_evidence_bindings"


@dataclass(slots=True)
class ScientificEvidenceStore:
    repository: StateRepository
    artifacts: ArtifactStore

    def register(self, binding: ScientificEvidenceBinding) -> None:
        self.resolve(binding)
        self.repository.append(EVIDENCE_BINDINGS, binding.to_dict())

    def resolve(self, binding: ScientificEvidenceBinding) -> dict:
        if binding.superseded_by:
            raise ValueError("scientific evidence binding is superseded")
        doc = read_verified_json(self.artifacts, binding.artifact_ref)
        if (doc.get("run_id") != binding.source_run_id or
                doc.get("lineage_id") != binding.lineage_id or
                doc.get("stage_name") != binding.stage_name or
                doc.get("kind") != binding.kind or not isinstance(doc.get("payload"), dict)):
            raise ValueError("scientific evidence source identity mismatch")
        return doc

    def superseded_refs(self, run_id: str, lineage_id: str) -> set[str]:
        latest = {str(row["binding_id"]): row for row in self.repository.all(EVIDENCE_BINDINGS)}
        return {str(row["artifact_ref"]) for row in latest.values() if row.get("target_run_id") == run_id
                and row.get("lineage_id") == lineage_id and row.get("superseded_by")}

    def supersede(self, ref: str, replacement_ref: str, *, run_id: str, lineage_id: str) -> None:
        old = read_verified_json(self.artifacts, ref)
        replacement = read_verified_json(self.artifacts, replacement_ref)
        if (old.get("lineage_id") != lineage_id or replacement.get("lineage_id") != lineage_id or
                old.get("stage_name") != replacement.get("stage_name") or ref == replacement_ref):
            raise ValueError("replacement scientific evidence identity mismatch")
        key = hashlib.sha256(canonical_bytes([run_id, ref])).hexdigest()[:24]
        tombstone = ScientificEvidenceBinding("superseded-"+key, run_id, str(old["run_id"]), lineage_id,
            str(old["stage_name"]), str(old["kind"]), ref, replacement_ref)
        new = ScientificEvidenceBinding("replacement-"+key, run_id, str(replacement["run_id"]), lineage_id,
            str(replacement["stage_name"]), str(replacement["kind"]), replacement_ref)
        # Invalidation precedes publication. An interrupted update has missing
        # evidence and fails closed; retrying publishes the same verified binding.
        self.repository.append(EVIDENCE_BINDINGS, tombstone.to_dict())
        self.register(new)

    def for_run(self, run_id: str, lineage_id: str) -> list[ScientificEvidenceBinding]:
        latest = {str(row["binding_id"]): row for row in self.repository.all(EVIDENCE_BINDINGS)}
        superseded = {str(row["artifact_ref"]) for row in latest.values() if row.get("superseded_by")
                      and row.get("target_run_id") == run_id and row.get("lineage_id") == lineage_id}
        return [ScientificEvidenceBinding.from_dict(row) for row in latest.values()
                if row.get("target_run_id") == run_id and row.get("lineage_id") == lineage_id
                and row["artifact_ref"] not in superseded]
