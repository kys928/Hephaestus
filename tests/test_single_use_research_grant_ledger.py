"""Signed research grant budget cannot silently fund more than the exact study."""
from __future__ import annotations

import pytest

from hephaestus.research.grant_usage import SingleUseComputeGrantLedger
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository


def _ledger(tmp_path):
    return SingleUseComputeGrantLedger(
        SQLiteStateRepository(tmp_path/"state.db"),
        FileSystemArtifactStore(tmp_path/"immutable"))


def _valid_report():
    return {"schema":"declared-CPU-fixture-not-real-GPU",
            "run_id":"research-a","lineage_id":"lineage-a",
            "role_invocations":6,"role_inference_scope":"genuine_certified_frozen_role_receipts",
            "status":"completed"}


def _request(ledger,fn,**kwargs):
    return ledger.run_once(grant_signature_sha256="sha256:"+"f"*64,
        run_id="research-a",lineage_id="lineage-a",
        maximum_calls=6,execute=fn,**kwargs)


def test_reserved_grant_cannot_repeat_paid_inference_on_retry(tmp_path):
    ledger=_ledger(tmp_path)
    calls=[]
    def completed():
        calls.append(1)
        return _valid_report()
    assert _request(ledger,completed)==_valid_report()
    assert _request(ledger,completed)==_valid_report()
    assert len(calls)==1
    rows=ledger.repository.all("research_signed_compute_grant_usage")
    assert [r["status"] for r in rows]==["reserved","completed"]


def test_interrupted_or_failed_study_requires_new_external_signed_grant(tmp_path):
    ledger=_ledger(tmp_path)
    def failed():
        raise RuntimeError("real generator died after partial inference")
    with pytest.raises(RuntimeError):
        _request(ledger,failed)
    with pytest.raises(PermissionError,match="new independent authorization"):
        _request(ledger,lambda:_valid_report())


def test_crash_after_reservation_never_reuses_existing_grant(tmp_path):
    ledger=_ledger(tmp_path)
    ledger.repository.append("research_signed_compute_grant_usage",{
        "grant_signature_sha256":"sha256:"+"f"*64,
        "run_id":"research-a","lineage_id":"lineage-a",
        "maximum_calls":6,"status":"reserved"})
    with pytest.raises(PermissionError,match="new independent authorization"):
        _request(ledger,lambda:_valid_report())


def test_incomplete_fake_report_cannot_consume_grant_as_success(tmp_path):
    ledger=_ledger(tmp_path)
    with pytest.raises(PermissionError,match="did not complete"):
        _request(ledger,lambda:{"status":"completed","role_invocations":0})
    assert ledger.repository.get_latest(
        "research_signed_compute_grant_usage",
        "grant_signature_sha256","sha256:"+"f"*64)["status"]=="failed_no_automatic_retry"


def test_grant_cross_run_replay_is_blocked_even_when_completed(tmp_path):
    ledger=_ledger(tmp_path)
    _request(ledger,lambda:_valid_report())
    with pytest.raises(PermissionError,match="different study"):
        ledger.run_once(grant_signature_sha256="sha256:"+"f"*64,
            run_id="other-run",lineage_id="lineage-a",maximum_calls=6,
            execute=lambda:_valid_report())



def test_compute_does_not_hold_state_writer_fence_during_long_inference(tmp_path):
    ledger=_ledger(tmp_path)
    def running():
        # A concurrent retry observes the durable reservation immediately;
        # it does NOT deadlock on a global write lock around GPU inference.
        with pytest.raises(PermissionError,match="new independent authorization"):
            _request(ledger,lambda:_valid_report())
        return _valid_report()
    assert _request(ledger,running)==_valid_report()
