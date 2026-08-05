"""Typed records for bounded infrastructure jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_JOB_STATUSES = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
    JobStatus.EXPIRED,
}


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    payload_ref: str
    owner_id: str
    run_id: str | None
    experiment_id: str | None
    idempotency_key: str | None
    status: JobStatus
    attempt_count: int
    created_at: datetime
    updated_at: datetime
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result_ref: str | None = None
    error_ref: str | None = None
    cancellation_requested: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        for key in (
            "created_at",
            "updated_at",
            "lease_expires_at",
            "started_at",
            "finished_at",
        ):
            value = payload[key]
            payload[key] = value.isoformat() if isinstance(value, datetime) else None
        return payload
