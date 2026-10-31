"""Durable provider intent; request secrets are never stored here."""
from dataclasses import dataclass
from ._base import JsonSchema


class ExternalOperationPending(RuntimeError):
    """A provider outcome must be reconciled before workflow execution continues."""

    def __init__(self, message: str, *, resumable: bool = True):
        super().__init__(message)
        self.resumable = resumable


@dataclass(slots=True)
class ExternalOperation(JsonSchema):
    operation_id: str
    provider: str
    request_sha256: str
    resource_name: str
    status: str
    external_id: str | None = None
    reconciliation_attempts: int = 0
