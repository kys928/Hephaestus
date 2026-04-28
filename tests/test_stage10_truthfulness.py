from pathlib import Path

from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.schemas.replay_metadata import build_replay_metadata
from hephaestus.state.run_store import RunStore


def test_replay_metadata_reference_level_is_explicitly_scoped() -> None:
    metadata = build_replay_metadata(
        checkpoint_ref="artifacts/run-1/model.ckpt",
        checkpoint_evidence={"integrity_level": "ref"},
    )

    assert metadata["requires_checkpoint_ref_match"] is True
    assert metadata["requires_content_hash_match"] is False
    assert metadata["content_hash_available"] is False
    assert metadata["replay_scope"] == "reference_only"
    limitations = [str(item) for item in metadata["limitations"]]
    assert any("content hash" in item for item in limitations)
    assert any("reference-level evidence" in item for item in limitations)


def test_replay_metadata_content_hash_is_only_claimed_with_hash_evidence() -> None:
    metadata = build_replay_metadata(
        checkpoint_ref="artifacts/run-2/model.ckpt",
        checkpoint_evidence={"integrity_level": "content_hash", "content_hash": "sha256:abc123"},
    )

    assert metadata["checkpoint_integrity_level"] == "content_hash"
    assert metadata["requires_content_hash_match"] is True
    assert metadata["content_hash_available"] is True
    assert metadata["replay_scope"] == "content_hash_verified"


def test_stage10_run_record_persists_truthful_replay_metadata(tmp_path: Path) -> None:
    run_id = "s10-truthful-run"
    build_orchestrator(state_root=tmp_path, run_id=run_id).run(run_id)

    record = RunStore(tmp_path).get(run_id)
    assert record is not None
    replay_metadata = dict(record.get("replay_metadata", {}))

    assert replay_metadata["requires_checkpoint_ref_match"] is True
    assert replay_metadata["requires_content_hash_match"] is False
    assert replay_metadata["content_hash_available"] is False
    assert replay_metadata["replay_scope"] == "reference_only"


def test_provenance_record_is_deferred_not_active_schema() -> None:
    repo = Path(__file__).resolve().parents[1]
    assert not (repo / "src/hephaestus/schemas/provenance_record.py").exists()

    src_text = "\n".join(path.read_text() for path in (repo / "src").rglob("*.py"))
    assert "ProvenanceRecord" not in src_text
