"""Materialize role context from verified scientific source bytes."""
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

from hephaestus.config_loader import load_named_config
from hephaestus.control.deterministic_role_boundary import (
    EvidenceRecord, EvidenceRegistry, DeterministicRoleContext,
)
from hephaestus.control.staged_state import StagedOperationRequest, WORKFLOW_RECORD_COLLECTION
from hephaestus.llm.role_contract import RoleTask, RoleEvidence
from hephaestus.schemas.scientific_evidence import ScientificEvidenceBinding
from hephaestus.state.evidence_store import ScientificEvidenceStore
from hephaestus.storage.verified import canonical_bytes


def evidence_excerpt(payload: dict, *, limit: int = 2900) -> str:
    """A bounded exact JSON excerpt; it asserts no interpretation of its values."""
    text = canonical_bytes(payload).decode()
    if len(text) <= limit:
        return text
    # Truncation is explicit, never turned into a claim of completeness.
    return "Source JSON prefix (truncated): " + text[:limit - 40]


@dataclass(slots=True)
class ArtifactModelEvidenceBinder:
    store: ScientificEvidenceStore
    config_dir: Path = Path("configs")
    maximum_sources: int = 8
    maximum_context_chars: int = 10_000

    def _workflow_sources(self, run_id: str, lineage_id: str):
        for row in self.store.repository.all(WORKFLOW_RECORD_COLLECTION):
            if row.get("run_id") != run_id or row.get("lineage_id") != lineage_id:
                continue
            if row.get("kind") in {"model_role_proposal", "approval_request", "approval_decision"}:
                continue
            ref = str(row.get("artifact_ref") or "")
            if not ref:
                raise ValueError("workflow source lacks immutable evidence receipt")
            binding = ScientificEvidenceBinding(str(row["record_id"]), run_id, run_id,
                lineage_id, str(row["stage_name"]), str(row["kind"]), ref)
            source = self.store.resolve(binding)
            expected = {k:v for k,v in row.items() if k != "artifact_ref"}
            if source != expected:
                raise ValueError("persisted workflow record differs from its immutable source")
            yield binding

    def carry_forward(self, source_run: str, target_run: str, lineage_id: str) -> None:
        """Explicitly retain source-run provenance when a new cycle starts."""
        rows = [*self.store.for_run(source_run, lineage_id),
                *self._workflow_sources(source_run, lineage_id)]
        for binding in rows:
            key = hashlib.sha256(canonical_bytes([target_run, binding.artifact_ref])).hexdigest()[:24]
            self.store.register(replace(binding, binding_id=f"binding-{key}", target_run_id=target_run))

    def bind(self, *, role: str, request: StagedOperationRequest):
        bindings = [*self.store.for_run(request.run_id, request.lineage_id),
                    *self._workflow_sources(request.run_id, request.lineage_id)]
        if not 1 <= self.maximum_sources <= 100 or not 1000 <= self.maximum_context_chars <= 12_000:
            raise ValueError("invalid model evidence context limits")
        # Validate all sources even when only a bounded subset fits the model.
        resolved = [(binding, self.store.resolve(binding)) for binding in bindings]
        unique = {binding.artifact_ref:(binding, source) for binding, source in resolved}
        selected = list(unique.values())[-self.maximum_sources:]
        per_source = min(2900, self.maximum_context_chars // max(1, len(selected)))
        records, evidence = {}, []
        for binding, source in selected:
            if binding.stage_name != request.stage_name:
                raise ValueError("evidence belongs to a different scientific stage")
            ref = binding.artifact_ref
            if ref in records:
                continue
            evidence.append(RoleEvidence(ref, evidence_excerpt(source, limit=per_source)))
            records[ref] = EvidenceRecord(ref, binding.kind, request.run_id, request.lineage_id,
                sha256=ref.removeprefix("sha256:"), observed_sha256=ref.removeprefix("sha256:"))
        if not evidence or len(evidence) > 100:
            raise ValueError("missing or excessive verified scientific evidence")
        stage = load_named_config(self.config_dir, "stage_profiles", request.stage_name)
        allowed = tuple(stage["allowed_next_actions"])
        if role == "controller":
            # Entry may request inspection/abort; it never grants a lineage mutation.
            allowed = ("request_recheck", "abort_run")
        context = DeterministicRoleContext(request.run_id, request.lineage_id, request.stage_name,
            stage_allowed_actions=allowed,
            verified_facts={"artifact_hash_match":True, "provenance_valid":True})
        task = RoleTask(role, f"Assess {len(selected)} of {len(unique)} verified source records for {request.phase}.{request.substep}. "
            "JSON values are recorded observations or explicitly labeled hypotheses; do not invent facts.",
            tuple(evidence), request.run_id, request.lineage_id, request.stage_name)
        return task, EvidenceRegistry(records), context
