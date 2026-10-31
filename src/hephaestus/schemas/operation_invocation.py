"""Durable intent/receipt for a staged service invocation."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class OperationInvocation(JsonSchema):
    workflow_id: str
    operation_id: str
    attempt: int
    status: str
    result_ref: str | None = None
    result: dict | None = None
