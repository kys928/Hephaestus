from pathlib import Path

from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.schemas.dataset_manifest import normalize_dataset_manifest
from hephaestus.state.manifest_store import ManifestStore
from hephaestus.state.run_store import RunStore


def test_full_manifest_normalization_is_complete() -> None:
    payload = {
        "manifest_id": "manifest-complete",
        "run_id": "run-complete",
        "lineage_id": "lineage-main",
        "stage_name": "early_pretraining",
        "artifact_ref": "artifacts/run-complete/dataset_manifest.json",
        "datasets": [
            {
                "dataset_id": "core-corpus-v2",
                "row_count": 1000,
                "version": "v2",
                "content_hash": "sha256:abc",
                "hash_type": "sha256",
                "source": "internal://core-corpus",
                "license": "internal",
            }
        ],
        "mixture_weights": {"core-corpus-v2": 1.0},
        "sampling_policy": {"kind": "fixed"},
    }

    manifest = normalize_dataset_manifest(payload)

    assert manifest["manifest_integrity_level"] == "complete"
    assert manifest["completeness_score"] == 1.0
    assert manifest["missing_fields"] == []


def test_reference_only_manifest_is_explicitly_incomplete() -> None:
    payload = {
        "manifest_id": "manifest-ref",
        "run_id": "run-ref",
        "lineage_id": "lineage-main",
        "artifact_ref": "artifacts/run-ref/dataset_manifest.json",
        "datasets": [{"source": "label-only:unknown"}],
    }

    manifest = normalize_dataset_manifest(payload)

    assert manifest["manifest_integrity_level"] in {"reference_only", "insufficient"}
    assert manifest["completeness_score"] < 1.0
    assert manifest["missing_fields"]
    assert manifest["warnings"]


def test_manifest_store_round_trip_preserves_strict_shape(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    payload = {
        "manifest_id": "manifest-store",
        "run_id": "run-store",
        "lineage_id": "lineage-store",
        "datasets": [{"dataset_id": "set-a", "row_count": 5, "version": "v1"}],
        "mixture_weights": {"set-a": 1.0},
    }
    store.append(payload)

    by_id = store.get("manifest-store")
    assert by_id is not None
    assert by_id["manifest_id"] == "manifest-store"
    assert "missing_fields" in by_id
    assert "warnings" in by_id

    by_run = store.list_for_run("run-store")
    assert len(by_run) == 1
    assert by_run[0]["datasets"][0]["dataset_id"] == "set-a"


def test_orchestrator_dry_run_persists_strict_manifest(tmp_path: Path) -> None:
    run_id = "manifest-dry-run"
    orchestrator = build_orchestrator(state_root=tmp_path, run_id=run_id)
    orchestrator.run(run_id)

    manifests = ManifestStore(tmp_path).list_for_run(run_id)
    assert manifests
    manifest = manifests[-1]

    assert manifest["run_id"] == run_id
    assert manifest["lineage_id"] == "lineage-main"
    assert manifest["stage_name"] == "early_pretraining"
    assert isinstance(manifest["datasets"], list)
    assert "completeness_score" in manifest
    assert "manifest_integrity_level" in manifest
    assert "missing_fields" in manifest
    assert "warnings" in manifest

    run_record = RunStore(tmp_path).all()[-1]
    assert run_record.get("data_manifest_id") == manifest["manifest_id"]


def test_backward_compatible_minimal_manifest_normalizes() -> None:
    legacy_payload = {
        "run_id": "legacy-run",
        "lineage_id": "legacy-lineage",
        "dataset_id": "legacy-set",
        "source_ids": ["legacy://set"],
        "total_examples": 10,
    }

    manifest = normalize_dataset_manifest(legacy_payload)

    assert manifest["manifest_id"] == "manifest-legacy-run"
    assert manifest["datasets"]
    assert manifest["datasets"][0]["dataset_id"] == "legacy-set"
    assert manifest["missing_fields"]
    assert manifest["warnings"]
