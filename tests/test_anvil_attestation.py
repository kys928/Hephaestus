from datetime import datetime, timedelta, timezone
from hashlib import sha256

import pytest

from hephaestus.control.anvil_attestation import AnvilAuthority
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes


def test_anvil_execution_grant_pins_task_worker_runtime_and_window(tmp_path):
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes_raw()
    authority = AnvilAuthority(public.hex(), "sha256:" + sha256(public).hexdigest())
    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")
    run = {"prepared_ref": "sha256:" + "2" * 64, "runtime_ref": "sha256:" + "3" * 64,
           "pod_id": "named-test-worker"}

    def issue(deadline):
        document = dict(run, schema="anvil.execution-grant.v1", device="cuda:0",
                        seed=271828, max_new_tokens=512, expires_at=deadline.isoformat())
        envelope = {"document": document, "signature_hex": private.sign(canonical_bytes(document)).hex()}
        return artifacts.put_bytes(canonical_bytes(envelope), media_type="application/json").artifact_ref

    now = datetime.now(timezone.utc)
    good = issue(now + timedelta(minutes=30))
    authority.verify_execution(artifacts, good, **run)
    with pytest.raises(PermissionError, match="different scope"):
        authority.verify_execution(artifacts, good, **dict(run, pod_id="other-worker"))
    with pytest.raises(PermissionError, match="expired"):
        authority.verify_execution(artifacts, issue(now - timedelta(seconds=1)), **run)
