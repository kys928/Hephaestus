"""Immutable CPU execution profile and operation cache in native SQLite custody."""
from hephaestus.policy.internal_research import put_record
from hephaestus.storage.verified import read_verified_json

PROFILES = "cpu_research_profiles"
RECEIPTS = "cpu_research_domain_receipts"


class CpuResearchState:
    def __init__(self, repository, artifacts, *, workflow_id, profile_ref):
        self.repository, self.artifacts = repository, artifacts
        self.workflow_id, self.profile_ref = workflow_id, profile_ref
        read_verified_json(artifacts, profile_ref)
        with repository.transaction() as tx:
            previous = repository.get_latest(PROFILES, "workflow_id", workflow_id)
            row = {"workflow_id": workflow_id, "profile_ref": profile_ref}
            if previous is None:
                tx.append(PROFILES, row)
            elif previous != row:
                raise PermissionError("CPU campaign execution profile changed on resume")

    def get(self, operation_id, input_ref):
        row = self.repository.get_latest(RECEIPTS, "operation_id", operation_id)
        if row is None:
            return None
        if (row.get("workflow_id") != self.workflow_id or row.get("profile_ref") != self.profile_ref
                or row.get("input_ref") != input_ref):
            raise PermissionError("CPU science operation identity/input changed on resume")
        return read_verified_json(self.artifacts, row["result_ref"])

    def put(self, operation_id, input_ref, payload):
        result_ref = put_record(self.artifacts, payload)
        with self.repository.transaction() as tx:
            existing = self.repository.get_latest(RECEIPTS, "operation_id", operation_id)
            row = {"workflow_id": self.workflow_id, "profile_ref": self.profile_ref,
                "operation_id": operation_id, "input_ref": input_ref, "result_ref": result_ref}
            if existing is None:
                tx.append(RECEIPTS, row)
            elif existing != row:
                raise PermissionError("CPU operation receipt is immutable")
        return result_ref
