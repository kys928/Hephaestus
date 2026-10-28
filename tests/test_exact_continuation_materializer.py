"""Synthetic caller-owned fixtures; never scientific rights/custody attestations."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from hephaestus.schemas.continuation_subset import ContinuationSubsetContract, PINNED_WIKITEXT_SUBSET
from hephaestus.data.continuation_materializer import reproduce_verified_bytes, digest, DATA_FORMAT
from hephaestus.data.continuation_eligibility import screen_continuation_rows
from hephaestus.data.english_heuristic import english_likelihood
from hephaestus.storage.verified import canonical_bytes


@pytest.fixture
def fixture(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"<|unk|>": 0, "the": 1, "alpha": 2, "beta": 3, "of": 4}, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok = tokenizer.to_str().encode()
    first = " ".join(["the", "alpha", "of", "beta"] * 56)  # 224 => two 160-token windows
    second = " ".join(["the", "beta"] * 80)  # 160 => one window
    rows = [{"text": ""}, {"text": first}, {"text": first},
            {"text": "the alpha alice@example.org"}, {"text": "the alpha"}, {"text": second}]
    report = screen_continuation_rows(rows, tokenizer, 0, lambda t: english_likelihood(t)[0])
    out = pa.BufferOutputStream(); pq.write_table(pa.Table.from_pylist(rows), out)
    raw = out.getvalue().to_pybytes()
    pack = canonical_bytes({"generation_probes": []})
    contract = ContinuationSubsetContract("Salesforce/wikitext", "a"*40, "train.parquet", digest(raw),
        len(raw), len(rows), "sha256:"+"b"*64, sha256(tok).hexdigest(), digest(pack),
        report["selection_commitment_sha256"], report["accepted_unique_rows"],
        report["eligible_tokens"], report["unknown_tokens"], report["eligible_within_row_windows"])
    return raw, tok, pack, contract, tokenizer, rows


def test_token_ids_source_positions_masks_and_no_retokenization(fixture):
    raw, tok, pack, spec, tokenizer, rows = fixture
    records, report = reproduce_verified_bytes(raw, tok, pack, spec)
    assert report["accepted_unique_rows"] == 2 and len(records) == 3
    assert [(r["source_row_index"], r["window_start_token"]) for r in records] == [(1,0),(1,64),(5,0)]
    for record in records:
        ids = tokenizer.encode(rows[record["source_row_index"]]["text"]).ids
        start = record["window_start_token"]
        assert record["input_ids"] == ids[start:start+160]
        assert record["labels"] == [-100]*128 + record["input_ids"][128:]
        assert record["token_count"] == 160 and "text" not in record
    assert reproduce_verified_bytes(raw, tok, pack, spec)[0] == records


@pytest.mark.parametrize("part", [0,1,2])
def test_byte_drift_fails_before_decoding(fixture, part):
    values = list(fixture[:3]); values[part] += b" "
    with pytest.raises(ValueError, match="SHA"):
        reproduce_verified_bytes(*values, fixture[3])


@pytest.mark.parametrize("change", [
    {"selection_commitment_sha256": "sha256:"+"d"*64},
    {"accepted_rows": 3}, {"eligible_tokens": 385}, {"windows": 4},
    {"stride_tokens": 32}, {"source_rows": 7},
])
def test_changed_inventory_fails_instead_of_silent_partial_output(fixture, change):
    raw, tok, pack, spec, *_ = fixture
    with pytest.raises(ValueError, match="inventory|commitment"):
        reproduce_verified_bytes(raw, tok, pack, replace(spec, **change))


def native_setup(tmp_path, fixture, *, trusted=True):
    from hephaestus.data.preprocessing import AutonomousDataPreprocessor, DataProcessingConfig
    from hephaestus.data.frozen_tokenizer import FrozenTokenizerCompatibilityChecker
    from hephaestus.data.schema_mapping import PinnedDatasetRecordMapper
    from hephaestus.schemas.dataset_mapping import DatasetRecordMapping
    from hephaestus.schemas.dataset_review import DatasetSourceReview, DatasetScreeningAccessScope
    from hephaestus.policy.dataset_review import (IndependentDatasetReviewPolicy, candidate_sha256,
        processing_profile_sha256, REFERENCES, AUDITS)
    from hephaestus.storage.filesystem import FileSystemArtifactStore
    from test_remote_data_preprocessing_bridge import _candidate, _selection, _approval, _receipt
    raw, tok, pack, spec, *_ = fixture
    source = tmp_path / "train.parquet"; source.write_bytes(raw)
    tokenizer = tmp_path / "tokenizer.json"; tokenizer.write_bytes(tok)
    public = tmp_path / "public.json"; public.write_bytes(pack)
    candidate = _candidate(); candidate.dataset_id = spec.dataset_id
    candidate.metadata["acquisition_files"] = [spec.source_file]
    selection = _selection(candidate); approval = _approval(candidate, selection)
    receipt = _receipt(source, candidate, selection, relative_path=spec.source_file)
    processor = AutonomousDataPreprocessor(DataProcessingConfig(tmp_path / "output"),
        tokenizer_checker=FrozenTokenizerCompatibilityChecker(tokenizer, spec.tokenizer_ref, spec.tokenizer_file_sha256),
        record_mapper=PinnedDatasetRecordMapper(DatasetRecordMapping([], "text", raw_text=True)),
        continuation_contract=spec)
    store = FileSystemArtifactStore(tmp_path / "fixture-attestations")
    def put(value):
        return store.put_bytes(canonical_bytes(value), media_type="application/json").artifact_ref
    refs = {name: put({"explicit_test_fixture_reference": name}) for name in REFERENCES}
    candidate_sha = candidate_sha256(candidate)
    access_ref = put(DatasetScreeningAccessScope(candidate_sha, [spec.source_file], 64*1024*1024,
        sorted(AUDITS), refs, status="authorized", authorized_by="synthetic-test-fixture").to_dict())
    hashes = {spec.source_file: spec.source_file_sha256}
    audits = {name: put({"audit": name, "status":"pass", "coverage":"full_corpus",
        "method":"synthetic unit test, not real clearance", "candidate_sha256":candidate_sha,
        "source_file_hashes":hashes, "reference_manifest_refs":refs, "access_scope_ref":access_ref}) for name in AUDITS}
    review = DatasetSourceReview(candidate_sha, "synthetic-fixture", "full_corpus", True,
        "pass","pass","pass","cleared",[put({"synthetic_rights":True})], spec.tokenizer_ref,
        spec.tokenizer_file_sha256,spec.eligible_tokens,spec.unknown_tokens,"original-screen-v1",
        source_file_hashes=hashes,audit_refs=audits,reference_manifest_refs=refs,access_scope_ref=access_ref,
        processing_profile_sha256=processing_profile_sha256(processor))
    review_ref = put(review.to_dict())
    policy = IndependentDatasetReviewPolicy(store,spec.tokenizer_ref,spec.tokenizer_file_sha256,
        frozenset({review_ref}) if trusted else frozenset(),frozenset({access_ref}))
    args = dict(run_id="fixture-run", lineage_id="fixture-lineage",stage_name="smoke_test",
        candidate=candidate,selection=selection,approval=approval,receipt=receipt,
        review_policy=policy,review_ref=review_ref,public_eval_pack_path=public)
    return processor,args


def test_native_exact_path_uses_existing_clearance_and_receipt(fixture,tmp_path):
    processor,args = native_setup(tmp_path,fixture)
    result = processor.process_reviewed_continuation_acquisition(**args)
    contract = result.trainable_data_contract
    assert contract.schema_version == DATA_FORMAT
    raw = Path(contract.processed_dataset_ref).read_bytes()
    assert digest(raw) == result.processed_content_hash
    assert len(raw.splitlines()) == 3
    assert result.processing_evidence["independent_review_ref"] == args["review_ref"]
    assert result.manifest.datasets[0]["row_count"] == 3
    assert result.manifest.chunking_policy["kind"] == "within_row_token_ids"
    assert processor.process_reviewed_continuation_acquisition(**args).processed_content_hash == result.processed_content_hash


def test_unaccepted_review_blocks_before_decoding_or_output(fixture,tmp_path,monkeypatch):
    processor,args = native_setup(tmp_path,fixture,trusted=False)
    def forbidden(*args,**kwargs):
        raise AssertionError("source decoded before review acceptance")
    monkeypatch.setattr("hephaestus.data.continuation_materializer.reproduce_verified_bytes",forbidden)
    with pytest.raises(PermissionError,match="has not been accepted"):
        processor.process_reviewed_continuation_acquisition(**args)
    assert not processor.config.artifact_root.exists()


def test_changed_contract_invalidates_previously_accepted_review(fixture,tmp_path):
    processor,args = native_setup(tmp_path,fixture)
    processor.continuation_contract = replace(processor.continuation_contract, stride_tokens=32)
    with pytest.raises(ValueError,match="profile"):
        processor.process_reviewed_continuation_acquisition(**args)
    assert not processor.config.artifact_root.exists()


def test_acquisition_approval_still_required_after_clearance(fixture,tmp_path):
    processor,args = native_setup(tmp_path,fixture)
    args["approval"] = replace(args["approval"], approval_refs=())
    with pytest.raises(PermissionError,match="approval"):
        processor.process_reviewed_continuation_acquisition(**args)
    assert not processor.config.artifact_root.exists()


def test_exact_path_never_falls_back_to_generic_word_chunks(fixture,tmp_path):
    processor,args = native_setup(tmp_path,fixture)
    allowed = {k:v for k,v in args.items() if k not in {"review_policy","review_ref","public_eval_pack_path"}}
    with pytest.raises(PermissionError,match="generic preprocessing"):
        processor.process_remote_acquisition(**allowed)


@pytest.mark.parametrize("change", [{"prefix_tokens":True},{"windows":0},{"source_file":"../train.parquet"},
    {"revision":"main"},{"unknown_tokens":True},{"unknown_tokens":20000}])
def test_contract_rejects_ambiguous_bounds(change):
    with pytest.raises(ValueError):
        replace(PINNED_WIKITEXT_SUBSET,**change)
