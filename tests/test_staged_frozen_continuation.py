"""Native exact-route integration with explicit synthetic clearance fixtures."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from hephaestus.control.staged_state import StagedAutonomousServices
from hephaestus.policy.dataset_review import candidate_sha256
from hephaestus.production.staged_finalization import StagedExperimentFinalizer
from hephaestus.backends.hf_causal_lm import directory_content_identity
from test_exact_continuation_materializer import fixture, native_setup
from test_staged_approved_local_data_factory import _write_row, _req
from test_staged_real_training_handoff import _runtime
from test_frozen_continuation_training import fixture as token_fixture, write_inputs


def staged_setup(tmp_path, fixture, *, trusted=True, configured=True, public=True):
    processor, args = native_setup(tmp_path, fixture, trusted=trusted)
    if not configured:
        processor.continuation_contract = None
    selection = args["selection"]
    selection.required_approvals = ["dataset_acquisition", "dataset_selection_approval",
                                    "dataset_review:" + args["review_ref"]]
    selection.metadata.update(dataset_review_ref=args["review_ref"],
                              candidate_sha256=candidate_sha256(args["candidate"]))
    runtime = _runtime(tmp_path)
    runtime.data_preprocessor = processor
    _write_row(runtime, "dataset_discovery", "dataset_discovery_result", {
        "request_id": selection.request_id, "status": "discovered",
        "candidates": [args["candidate"].to_dict()]})
    _write_row(runtime, "dataset_selection", "dataset_selection_decision", selection.to_dict())
    approval_ref = args["approval"].approval_refs[0]
    _write_row(runtime, "acquisition_approval_gate", "approval_decision", {
        "subject_ref": selection.decision_id, "status": "approved",
        "approval_ref": approval_ref, "requirements": selection.required_approvals})
    _write_row(runtime, "dataset_acquisition", "acquisition_receipt", args["receipt"].to_dict())
    service = runtime.with_huggingface_data_factory(StagedAutonomousServices(),
        review_policy=args["review_policy"], acquisition_enabled=True,
        public_eval_pack_path=args["public_eval_pack_path"] if public else None).data_preprocessor
    req = replace(_req("manifest_production"), prior_outputs={
        "data_acquisition_audit.acquisition_approval_gate": {
            "approval_status": "approved", "approval_ref": approval_ref}})
    return service, req, processor


def test_native_staged_route_materializes_only_explicit_reviewed_tokens(fixture, tmp_path):
    service, req, processor = staged_setup(tmp_path, fixture)
    result = service.execute(req)
    assert result.status == "completed", result.blocking_issues
    factory = next(row.payload for row in result.records if row.kind == "data_factory_result")
    assert factory["trainable_data_contract"]["schema_version"] == "trainable-token-ids.v1"
    rows = [json.loads(line) for line in Path(factory["processed_dataset_ref"]).read_text().splitlines()]
    assert len(rows) == 3 and all("text" not in row for row in rows)
    assert factory["processing_evidence"]["tokenizer_compatibility"]["status"] == "checked"


@pytest.mark.parametrize("options", [{"trusted": False}, {"configured": False}, {"public": False}])
def test_staged_route_cannot_use_unaccepted_review_or_implicit_fallback(fixture, tmp_path, options):
    service, req, processor = staged_setup(tmp_path, fixture, **options)
    result = service.execute(req)
    assert result.status == "blocked"
    assert not processor.config.artifact_root.exists()


@pytest.mark.parametrize("fault", [None, "different_backend", "token_budget", "context_budget"])
def test_finalizer_binds_artifact_identity_to_separate_pinned_loader_location(tmp_path, monkeypatch, fault):
    from test_staged_verified_training_finalization import _setup_finalized

    runtime, finalizer, request, _ = _setup_finalized(tmp_path)
    old_data = finalizer._data(request)
    original = finalizer._load(request, "experiment_proposal", "experiment_proposal")
    model = finalizer._model(request)[1]
    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    backend, records, evidence, _ = token_fixture(frozen_root)
    evidence["tokenizer_compatibility"]["status"] = "checked"
    compatibility, constraints = write_inputs(frozen_root, records, evidence)
    tokenizer_dir = Path(model.tokenizer_ref)
    if fault == "different_backend":
        backend.add_tokens(["changed"])
    (tokenizer_dir / "tokenizer.json").write_text(backend.to_str())
    original["training_constraints"].update(tokenizer_revision=directory_content_identity(tokenizer_dir),
        vocabulary_size=8, context_length=159 if fault == "context_budget" else 160,
        max_total_tokens=319 if fault == "token_budget" else 320)
    load = StagedExperimentFinalizer._load
    monkeypatch.setattr(StagedExperimentFinalizer, "_load", lambda self, req, step, kind:
        original if step == "experiment_proposal" else load(self, req, step, kind))
    # The upstream authorization/immutable-record tests separately cover _data;
    # this fixture isolates tokenizer artifact-vs-loader binding in finalize.
    replacement = (*old_data[:2], evidence, Path(compatibility["processed_dataset_ref"]),
                   Path(constraints["processing_evidence_ref"]),
                   Path(compatibility["trainable_data_contract_ref"]),
                   evidence["tokenizer_compatibility"], old_data[-1])
    monkeypatch.setattr(StagedExperimentFinalizer, "_data", lambda self, req: replacement)
    if fault:
        with pytest.raises(ValueError, match="tokenizer location|recipe cannot fit"):
            finalizer.finalize(request)
    else:
        result = finalizer.finalize(request)
        assert result.training_constraints["tokenizer_id"] == str(tokenizer_dir)
        assert result.training_constraints["tokenizer_id"] != evidence["tokenizer_compatibility"]["tokenizer_ref"]
