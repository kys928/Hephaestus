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


def test_signed_unresolved_terminal_is_durable_and_closes_plan(tmp_path):
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from hephaestus.control.anvil_context import AnvilAdmission
    from hephaestus.control.anvil_dispatch import AnvilDispatch
    from hephaestus.control.anvil_execution import _require_open, retain_terminal
    from hephaestus.jobs import SQLiteJobQueue
    from hephaestus.state.anvil import AnvilState
    from hephaestus.storage.sqlite import SQLiteStateRepository

    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes_raw()
    authority = AnvilAuthority(public.hex(), "sha256:" + sha256(public).hexdigest())
    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")
    repository = SQLiteStateRepository(tmp_path / "state.sqlite")
    board = AnvilState(repository, lambda _: False, lambda *_: False)

    def publish(payload):
        return artifacts.put_bytes(canonical_bytes(payload), media_type="application/json").artifact_ref

    registry_ref = "sha256:" + "1" * 64
    plan_ref = publish({
        "schema": "anvil.dispatch.v1", "investigation": "trial", "run_id": "trial-run",
        "lineage_id": "trial-lineage", "registry_ref": registry_ref,
        "material_refs": ["sha256:" + char * 64 for char in "234"],
    })
    dispatch = AnvilDispatch(AnvilAdmission(board, artifacts, registry_ref, frozenset(), frozenset()),
                             SQLiteJobQueue(tmp_path / "queue.sqlite"), plan_ref)
    frontier = board.read_snapshot("trial").frontier
    review_ref = publish({"reason": "No admissible outputs"})
    document = {
        "schema": "anvil.terminal.v1", "execution_authorized": False, "plan_ref": plan_ref,
        "frontier": frontier, "outcome": "unresolved", "review_ref": review_ref,
    }
    terminal_ref = publish({
        "document": document, "signature_hex": private.sign(canonical_bytes(document)).hex(),
    })
    written = retain_terminal(dispatch, terminal_ref=terminal_ref, frontier=frontier,
                              authority=authority, repository=repository)
    restarted = SQLiteStateRepository(tmp_path / "state.sqlite")
    assert retain_terminal(dispatch, terminal_ref=terminal_ref, frontier=frontier,
                           authority=authority, repository=restarted) == written
    with pytest.raises(PermissionError, match="terminal review"):
        _require_open(restarted, plan_ref)
