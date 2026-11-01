"""External one-use authority interface; no production keys or self-authorization.

The institution must operate the implementation independently of campaign backup
and rollback and configure its key/namespace out of band. A signature alone does
not establish that independence. This module does not implement that service.
"""
from datetime import datetime, timezone
from typing import Protocol

from hephaestus.schemas.exposure_ledger import ExposureReservation, ExposureReceipt
from hephaestus.storage.verified import canonical_bytes


class ExposureLedger(Protocol):
    namespace: str
    authority_scope: str

    def consume_once(self, reservation: dict, grant: dict, *, submission: dict) -> dict:
        """Atomically spend material globally, then sign only the first response.

        Verify that the grant's submission binds the requested material; enforce
        uniqueness of material, grant and study across changed IDs/nonces. Do not reissue a
        generation permission on retries, reads, timeouts or local restoration.
        Unknown outcomes are consumed/blocked until independent adjudication.
        """
        ...


def verify_exposure_receipt(envelope, *, reservation, trusted_keys, namespace, authority_scope):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from hephaestus.evaluation.prospective_pipeline import digest
    try:
        ExposureReservation.from_dict(reservation)
        payload=envelope['payload']; record=ExposureReceipt.from_dict(payload)
        if (set(envelope)!={'payload','signature_hex'} or set(payload)!=set(record.to_dict())
                or record.namespace!=namespace or record.authority_scope!=authority_scope
                or record.request_commitment!=digest(reservation)
                or record.material_commitment!=reservation['material_commitment']):
            raise ValueError('external receipt scope differs')
        when=datetime.fromisoformat(record.issued_at)
        if when.utcoffset() is None or when>datetime.now(timezone.utc):raise ValueError('invalid authority timestamp')
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted_keys[record.authority_id])).verify(
            bytes.fromhex(envelope['signature_hex']),canonical_bytes(payload))
        if reservation['purpose']=='confirmatory' and authority_scope!='entrusted-external-service':
            raise ValueError('fixture authority cannot permit confirmation')
    except Exception as exc:
        raise PermissionError('external exposure receipt is not entrusted and exactly scoped') from exc
    return digest(envelope)
