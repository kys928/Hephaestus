"""Authenticate independent Anvil execution, receipt and terminal evidence."""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256

from hephaestus.schemas.anvil import require_digest
from hephaestus.storage.verified import canonical_bytes, read_verified_json


@dataclass(frozen=True, slots=True)
class AnvilAuthority:
    public_key_hex: str
    public_key_ref: str

    def verify(self, artifacts, ref: str, *, schema: str, expected: dict) -> dict:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        key = bytes.fromhex(self.public_key_hex)
        if len(key) != 32 or "sha256:" + sha256(key).hexdigest() != self.public_key_ref:
            raise PermissionError("independently pinned Anvil authority differs")
        envelope = read_verified_json(artifacts, ref, limit=8192)
        if set(envelope) != {"document", "signature_hex"} or not isinstance(envelope["document"], dict):
            raise PermissionError("invalid Anvil signature envelope")
        document = envelope["document"]
        try:
            signature = bytes.fromhex(envelope["signature_hex"])
            Ed25519PublicKey.from_public_bytes(key).verify(signature, canonical_bytes(document))
        except (TypeError, ValueError, InvalidSignature) as error:
            raise PermissionError("invalid independent Anvil signature") from error
        if document.get("schema") != schema or any(document.get(k) != v for k, v in expected.items()):
            raise PermissionError("Anvil signature binds a different scope")
        return document

    def verify_execution(self, artifacts, ref: str, *, prepared_ref: str, pod_id: str, runtime_ref: str) -> None:
        require_digest(prepared_ref)
        require_digest(runtime_ref)
        if not pod_id:
            raise PermissionError("a named provider worker is required")
        document = self.verify(artifacts, ref, schema="anvil.execution-grant.v1", expected={
            "prepared_ref": prepared_ref, "pod_id": pod_id, "runtime_ref": runtime_ref,
            "device": "cuda:0", "max_new_tokens": 512, "seed": 271828,
        })
        try:
            deadline = datetime.fromisoformat(document["expires_at"])
        except (KeyError, TypeError, ValueError) as error:
            raise PermissionError("Anvil grant requires an execution deadline") from error
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds() if deadline.tzinfo else -1
        if not 0 < remaining <= 3600:
            raise PermissionError("Anvil execution grant expired or exceeds the investigation window")
