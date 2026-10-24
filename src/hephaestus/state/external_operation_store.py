"""Append-only external-operation identities using the existing repository."""
from dataclasses import dataclass
from hephaestus.schemas.external_operation import ExternalOperation
from hephaestus.storage.base import StateRepository

EXTERNAL_OPERATIONS = "external_operations"


@dataclass(slots=True)
class ExternalOperationStore:
    repository: StateRepository

    def get(self, operation_id: str) -> ExternalOperation | None:
        row = self.repository.get_latest(EXTERNAL_OPERATIONS, "operation_id", operation_id)
        return ExternalOperation.from_dict(row) if row is not None else None

    def record(self, operation: ExternalOperation) -> None:
        if operation.status not in {"submitting", "uncertain", "accepted", "blocked", "terminating", "terminated"}:
            raise ValueError("invalid external operation state")
        prior = self.get(operation.operation_id)
        if prior is not None:
            if (prior.provider, prior.request_sha256, prior.resource_name) != (
                    operation.provider, operation.request_sha256, operation.resource_name):
                raise ValueError("external operation identity cannot be changed")
            if prior.external_id and prior.external_id != operation.external_id:
                raise ValueError("external operation cannot change its allocated resource")
        self.repository.append(EXTERNAL_OPERATIONS, operation.to_dict())

    def for_resource(self, external_id: str) -> ExternalOperation | None:
        row = self.repository.get_latest(EXTERNAL_OPERATIONS, "external_id", external_id)
        return ExternalOperation.from_dict(row) if row is not None else None
