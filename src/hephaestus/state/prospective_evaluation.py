"""Append-only exposure journal. Original historical references are never created here."""
from hephaestus.storage.verified import canonical_bytes, read_verified_json

COLLECTION = "internal_prospective_evaluation"


class ProspectiveEvaluationStore:
    def __init__(self, repository, artifacts):
        self.repository, self.artifacts = repository, artifacts

    def freeze(self, *, experiment_id, evaluation_ref, development_ref, method,
               limitations):
        if not experiment_id or not method or not limitations:
            raise ValueError("evaluation method and honest limitations are required")
        # One protocol per experiment; repeat calls cannot rewrite or recover absent history.
        with self.repository.transaction() as tx:
            if any(r.get("experiment_id") == experiment_id for r in self.repository.all(COLLECTION)):
                raise PermissionError("experiment protocol/exposure already exists")
            for ref in (evaluation_ref, development_ref):
                data = read_verified_json(self.artifacts, ref)
                if not isinstance(data.get("examples"), list) or not data["examples"]:
                    raise ValueError("original prospective examples required")
            payload = {"schema": "internal-prospective-evaluation.v1",
                "experiment_id": experiment_id, "evaluation_ref": evaluation_ref,
                "development_ref": development_ref, "method": method,
                "limitations": list(limitations), "external_certification": False,
                "historical_reference": False}
            ref = self.artifacts.put_bytes(canonical_bytes(payload), media_type="application/json").artifact_ref
            tx.append(COLLECTION, {"experiment_id": experiment_id, "event": "freeze", "protocol_ref": ref})
        return ref

    def require_frozen(self, experiment_id, protocol_ref):
        rows = [r for r in self.repository.all(COLLECTION) if r.get("experiment_id") == experiment_id]
        if not rows or rows[0] != {"experiment_id": experiment_id, "event": "freeze", "protocol_ref": protocol_ref}:
            raise PermissionError("prospective evaluation must be frozen before exposure")
        protocol = read_verified_json(self.artifacts, protocol_ref)
        if (protocol.get("schema") != "internal-prospective-evaluation.v1" or
                protocol.get("experiment_id") != experiment_id or
                protocol.get("external_certification") is not False or
                protocol.get("historical_reference") is not False):
            raise PermissionError("internal evaluation scope changed")
        for key in ("evaluation_ref", "development_ref"):
            read_verified_json(self.artifacts, protocol[key])
        return protocol

    def record_exposure(self, experiment_id, protocol_ref, source_ref):
        with self.repository.transaction() as tx:
            self.require_frozen(experiment_id, protocol_ref)
            read_verified_json(self.artifacts, source_ref)
            tx.append(COLLECTION, {"experiment_id": experiment_id, "event": "training_exposure",
                                   "protocol_ref": protocol_ref, "source_ref": source_ref})
