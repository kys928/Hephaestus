"""Durable evidence references, without caller-authored scientific summaries."""
from dataclasses import dataclass
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

    def for_run(self, run_id: str, lineage_id: str) -> list[ScientificEvidenceBinding]:
        latest = {str(row["binding_id"]): row for row in self.repository.all(EVIDENCE_BINDINGS)}
        return [ScientificEvidenceBinding.from_dict(row) for row in latest.values()
                if row.get("target_run_id") == run_id and row.get("lineage_id") == lineage_id]
