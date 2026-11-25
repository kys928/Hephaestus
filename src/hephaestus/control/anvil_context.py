"""Deterministic context preparation and cold admission of frozen-role receipts."""
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Protocol

from hephaestus.interfaces.anvil import Blackboard
from hephaestus.llm.inference_receipts import verify_inference_receipt
from hephaestus.llm.role_contract import RoleEvidence, RoleTask
from hephaestus.production.original_cuda_identity import verified_original_roles
from hephaestus.schemas.anvil import Contribution, ContributionKind, AcceptedContribution, ContributionOrigin, require_digest
from hephaestus.storage.base import ArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json


class MaterialLedger(Protocol):
    def all(self, collection: str) -> list[dict[str, object]]: ...
    def append(self, collection: str, payload: dict[str, object], *, operation_id: str): ...


def material_collection(origin: ContributionOrigin) -> str:
    scope = [origin.investigation, origin.run_id, origin.lineage_id]
    return "anvil.material." + sha256(canonical_bytes(scope)).hexdigest()[:40]


def read_scoped_material(artifacts: ArtifactStore, origin: ContributionOrigin, ref: str) -> dict:
    doc = read_verified_json(artifacts, ref, limit=8192)
    if (doc.get("schema") != "anvil.material.v1"
            or (doc.get("investigation"), doc.get("run_id"), doc.get("lineage_id"))
            != (origin.investigation, origin.run_id, origin.lineage_id)
            or doc.get("kind") not in {"objective", "evidence", "experiment_result"}
            or not isinstance(doc.get("text"), str) or not 1 <= len(doc["text"]) <= 500):
        raise ValueError("invalid scoped material")
    return doc


@dataclass(frozen=True, slots=True)
class AnvilAdmission:
    _board: Blackboard
    artifacts: ArtifactStore
    registry_ref: str
    trusted_material: frozenset[str]
    trusted_receipts: frozenset[str]
    _material_ledger: MaterialLedger | None = None

    def admit_material(self, origin: ContributionOrigin, ref: str) -> None:
        if self._material_ledger is None or ref not in self.trusted_material:
            raise PermissionError("material lacks independent admission")
        doc = read_scoped_material(self.artifacts, origin, ref)
        self._material_ledger.append(
            material_collection(origin), {"ref": ref, "kind": doc["kind"]}, operation_id=ref,
        )


    def prepare(self, origin: ContributionOrigin, sequence: int, material_refs: tuple[str, ...]) -> RoleTask:
        snapshot = self._board.read_snapshot(origin.investigation, sequence=sequence)
        if snapshot.frontier != origin.context_frontier:
            raise PermissionError("context frontier differs from retained snapshot")
        if not 1 <= len(material_refs) <= 8 or len(set(material_refs)) != len(material_refs):
            raise ValueError("expected one to eight distinct trusted material references")
        retained = (
            {row["ref"] for row in self._material_ledger.all(material_collection(origin))}
            if self._material_ledger is not None else self.trusted_material
        )
        evidence = []
        objectives = 0
        for ref in sorted(material_refs):
            if ref not in retained:
                raise PermissionError("material lacks independent admission")
            doc = read_scoped_material(self.artifacts, origin, ref)
            objectives += doc["kind"] == "objective"
            evidence.append(RoleEvidence(ref, f'{doc["kind"]}: {doc["text"]}'))
        if objectives != 1:
            raise ValueError("context requires exactly one trusted objective")
        scoped = [
            row for row in snapshot.contributions
            if (row.origin.run_id, row.origin.lineage_id) == (origin.run_id, origin.lineage_id)
        ]
        rows = []
        for row in scoped[-8:]:
            entry = {"id": row.contribution_id, "role": row.origin.role, "kind": row.payload.kind,
                     "text_excerpt": row.payload.text[:350], "receipt": row.origin.inference_ref}
            if row.payload.v1_position is not None:
                entry.update(v1_position=row.payload.to_dict()["v1_position"],
                             confidence=row.payload.confidence, evidence_refs=list(row.payload.evidence_refs),
                             uncertainties=[item[:100] for item in row.payload.uncertainties[:2]])
            while rows and len(canonical_bytes(rows + [entry])) > 10_000:
                rows.pop(0)
            rows.append(entry)
        situation = canonical_bytes({
            "investigation": origin.investigation, "frontier": snapshot.frontier,
            "advisory_history": rows,
        }).decode()
        return RoleTask(origin.role, situation, tuple(evidence), origin.run_id, origin.lineage_id, "anvil_rehearsal")

    def accept(self, origin: ContributionOrigin, sequence: int, material_refs: tuple[str, ...]) -> AcceptedContribution:
        stack, contract = verified_original_roles(self.registry_ref)
        require_digest(origin.inference_ref)
        if origin.inference_ref not in self.trusted_receipts:
            raise PermissionError("receipt lacks independent admission")
        model = stack.resolve(origin.role)
        task = self.prepare(origin, sequence, material_refs)
        receipt = verify_inference_receipt(self.artifacts, origin.inference_ref, contract)
        expected = {
            "status": "accepted", "role": origin.role, "run_id": origin.run_id,
            "lineage_id": origin.lineage_id, "stage_name": task.stage_name, "prompt": contract.prompt(task),
            "model_id": model.model_id, "base_revision": model.revision,
            "adapter_sha256": model.adapter.sha256, "adapter_source": model.adapter.s3_key,
            "tokenizer_id": model.model_id, "tokenizer_revision": model.revision,
            "evidence_refs": [item.ref for item in task.evidence],
        }
        if (origin.model_id, origin.model_revision, origin.adapter_ref) != (
            model.model_id, model.revision, "sha256:" + model.adapter.sha256
        ) or any(receipt.get(key) != value for key, value in expected.items()):
            raise PermissionError("receipt does not bind the pinned role and exact context")
        answer = receipt["answer"]
        accepted_origin = replace(origin, idempotency_key=origin.inference_ref)
        return self._board.append_contribution(accepted_origin, Contribution.from_model({
            "kind": ContributionKind.MODEL_CLAIM,
            "text": answer["rationale"],
            "v1_position": {key: answer[key] for key in ("decision", "action", "primary_variable")},
            "evidence_refs": answer["evidence_refs"],
            "confidence": answer["confidence"],
            "uncertainties": answer["uncertainties"],
        }))
