"""Explicit grant, development-attempt and durable session records."""
from dataclasses import dataclass, field
import re
from ._base import JsonSchema


@dataclass(slots=True)
class CustodyGrant(JsonSchema):
    study_id: str
    purpose: str
    allowed_sessions: int
    submission_commitment: str
    authority_id: str
    issued_at: str
    expires_at: str

    def __post_init__(self):
        if (not self.study_id or not self.authority_id or self.purpose not in {"confirmatory", "rehearsal"}
                or type(self.allowed_sessions) is not int or self.allowed_sessions != 1
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.submission_commitment)):
            raise ValueError("invalid custody grant scope")


@dataclass(slots=True)
class DevelopmentAttempt(JsonSchema):
    seed: int
    status: str
    development_correct: int = 0
    development_valid_json: int = 0
    error_ref: str | None = None

    def __post_init__(self):
        if (type(self.seed) is not int or self.status not in {"completed", "failed"}
                or any(type(v) is not int or v < 0 for v in (self.development_correct, self.development_valid_json))
                or self.development_correct > self.development_valid_json
                or (self.status == "failed" and not self.error_ref)):
            raise ValueError("invalid or unsupported development attempt evidence")


@dataclass(slots=True)
class CustodySession(JsonSchema):
    study_id: str
    submission_commitment: str
    submission_ref: str
    grant_ref: str
    started_at: str
    status: str = "consumed"
    bundle_ref: str | None = None
    error_type: str | None = None
    partial_generation_refs: dict = field(default_factory=dict)
    exposure_request_ref: str | None = None
    exposure_receipt_ref: str | None = None
    exposure_authority_scope: str | None = None
    partial_raw_index_ref: str | None = None
