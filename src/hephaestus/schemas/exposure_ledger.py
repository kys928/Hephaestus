"""One irreversible assessment reservation, retained outside campaign backups."""
from dataclasses import dataclass
import re
from uuid import UUID
from ._base import JsonSchema


@dataclass(slots=True)
class ExposureReservation(JsonSchema):
    study_id: str
    purpose: str
    submission_commitment: str
    material_commitment: str
    grant_ref: str
    attempt_nonce: str
    schema: str = 'external-exposure-reservation.v1'

    def __post_init__(self):
        if (not self.study_id or self.purpose not in {'confirmatory','rehearsal'}
                or self.schema != 'external-exposure-reservation.v1'
                or any(not re.fullmatch(r'sha256:[0-9a-f]{64}',x) for x in
                       (self.submission_commitment,self.material_commitment,self.grant_ref))
                or str(UUID(self.attempt_nonce)) != self.attempt_nonce):
            raise ValueError('invalid independent exposure reservation')


@dataclass(slots=True)
class ExposureReceipt(JsonSchema):
    authority_id: str
    namespace: str
    authority_scope: str
    request_commitment: str
    material_commitment: str
    monotonic_sequence: int
    issued_at: str
    status: str = 'permanently-consumed'
    schema: str = 'external-exposure-receipt.v1'

    def __post_init__(self):
        if (not self.authority_id or not self.namespace
                or self.authority_scope not in {'nonconfirmatory-fixture','entrusted-external-service'}
                or type(self.monotonic_sequence) is not int or self.monotonic_sequence < 1
                or self.status != 'permanently-consumed' or self.schema != 'external-exposure-receipt.v1'
                or any(not re.fullmatch(r'sha256:[0-9a-f]{64}',x) for x in
                       (self.request_commitment,self.material_commitment))):
            raise ValueError('invalid irreversible exposure receipt')


@dataclass(slots=True)
class RawSampleReceiptLink(JsonSchema):
    study_id: str
    model_label: str
    sample_id: str
    raw_sample_ref: str
    schema: str = 'custody-raw-sample-link.v1'
