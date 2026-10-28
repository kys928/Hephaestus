"""Deterministic exact-token reproduction and governed native materialization.

Reproduction is an in-memory transform, never an approval. Only the native
entrypoint writes training inputs, after independently accepted clearance and
verification of an existing acquisition receipt. Generic WikiText stays blocked.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
import tempfile
from pathlib import Path

from hephaestus.data.continuation_eligibility import screen_continuation_rows
from hephaestus.data.english_heuristic import english_likelihood
from hephaestus.schemas.continuation_subset import ContinuationSubsetContract
from hephaestus.storage.verified import canonical_bytes

DATA_FORMAT = "trainable-token-ids.v1"


def digest(raw: bytes) -> str:
    return "sha256:" + sha256(raw).hexdigest()


def public_prompts(raw: bytes, contract: ContinuationSubsetContract) -> tuple[str, ...]:
    if digest(raw) != contract.public_eval_pack_sha256:
        raise ValueError("frozen public evaluation pack SHA mismatch")
    pack = json.loads(raw)
    return tuple(p["prompt"] for field in (
        "generation_probes", "continuation_prompts", "structure_tests",
        "repetition_checks", "length_termination_checks", "regression_prompts",
    ) for p in pack.get(field, []) if isinstance(p.get("prompt"), str))


def token_windows(position: int, content: str, ids: list[int],
                  contract: ContinuationSubsetContract):
    width = contract.prefix_tokens + contract.target_tokens
    content_hash = digest(content.encode("utf-8"))
    for start in range(0, len(ids) - width + 1, contract.stride_tokens):
        window = ids[start:start + width]
        yield {
            "record_kind": "frozen_token_continuation",
            "source_row_index": position,
            "source_content_sha256": content_hash,
            "window_start_token": start,
            "token_count": width,
            "input_ids": window,
            "labels": [-100] * contract.prefix_tokens + window[contract.prefix_tokens:],
        }


def verify_inventory(report: dict, contract: ContinuationSubsetContract) -> None:
    expected = {
        "source_rows_examined": contract.source_rows,
        "accepted_unique_rows": contract.accepted_rows,
        "eligible_tokens": contract.eligible_tokens,
        "unknown_tokens": contract.unknown_tokens,
        "eligible_within_row_windows": contract.windows,
        "selection_commitment_sha256": contract.selection_commitment_sha256,
        "within_row_prefix_tokens": contract.prefix_tokens,
        "within_row_target_tokens": contract.target_tokens,
        "within_row_stride_tokens": contract.stride_tokens,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("reproduced subset differs from exact reviewed inventory or commitment")


def reproduce_windows(rows: list[dict], tokenizer, contract: ContinuationSubsetContract,
                      prompts: tuple[str, ...]) -> tuple[list[dict], dict]:
    """Pure transform for already authorized caller-owned bytes; never writes files."""
    contract.validate()
    unknown = tokenizer.token_to_id("<|unk|>")
    if unknown is None:
        raise ValueError("frozen tokenizer unknown ID missing")
    records = []
    report = screen_continuation_rows(
        rows, tokenizer, unknown, lambda text: bool(english_likelihood(text)[0]),
        public_prompts=prompts, prefix_tokens=contract.prefix_tokens,
        target_tokens=contract.target_tokens, stride_tokens=contract.stride_tokens,
        accepted_row_observer=lambda i, text, ids: records.extend(token_windows(i, text, ids, contract)),
    )
    verify_inventory(report, contract)
    if len(records) != contract.windows:
        raise ValueError("materialized token-window count differs from reviewed count")
    return records, report


def reproduce_verified_bytes(source: bytes, tokenizer_bytes: bytes, pack_bytes: bytes,
                             contract: ContinuationSubsetContract) -> tuple[list[dict], dict]:
    """Authenticate the same byte buffers that are decoded; no path reopen race."""
    if len(source) != contract.source_bytes or digest(source) != contract.source_file_sha256:
        raise ValueError("exact source file bytes/SHA mismatch")
    if sha256(tokenizer_bytes).hexdigest() != contract.tokenizer_file_sha256:
        raise ValueError("original frozen tokenizer bytes/SHA mismatch")
    prompts = public_prompts(pack_bytes, contract)
    from tokenizers import Tokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(pa.BufferReader(source))
    if table.num_rows != contract.source_rows or table.column_names != ["text"]:
        raise ValueError("exact source row inventory or schema mismatch")
    return reproduce_windows(table.to_pylist(), Tokenizer.from_str(tokenizer_bytes.decode("utf-8")), contract, prompts)


def materialize_approved_continuation(preprocessor, *, run_id: str, lineage_id: str,
        stage_name: str, candidate, selection, approval, receipt,
        review_policy, review_ref: str, public_eval_pack_path: Path):
    """Explicit native preprocessing boundary; no source I/O before clearance."""
    from hephaestus.policy.dataset_review import IndependentDatasetReviewPolicy, processing_profile_sha256
    from hephaestus.data.materialization import validate_remote_acquisition_for_preprocessing
    from hephaestus.data.frozen_tokenizer import FrozenTokenizerCompatibilityChecker
    from hephaestus.data.schema_mapping import PinnedDatasetRecordMapper
    from hephaestus.data.preprocessing import DataFactoryResult
    from hephaestus.data.manifest_builder import build_dataset_manifest
    from hephaestus.data.contract_builder import build_preprocessing_contracts
    from hephaestus.utils.io import write_json

    contract = preprocessor.continuation_contract
    if not isinstance(contract, ContinuationSubsetContract):
        raise PermissionError("explicit exact continuation contract required")
    contract.validate()
    if (candidate.provider_id != "huggingface" or candidate.dataset_id != contract.dataset_id
            or candidate.revision != contract.revision
            or candidate.metadata.get("acquisition_files") != [contract.source_file]):
        raise ValueError("candidate differs from exact continuation source contract")
    checker = preprocessor.tokenizer_checker
    mapper = preprocessor.record_mapper
    if (not isinstance(checker, FrozenTokenizerCompatibilityChecker)
            or checker.tokenizer_ref != contract.tokenizer_ref
            or checker.expected_file_sha256 != contract.tokenizer_file_sha256
            or not isinstance(mapper, PinnedDatasetRecordMapper)
            or mapper.mapping.target_field != "text" or not mapper.mapping.raw_text
            or mapper.mapping.prompt_fields or preprocessor.record_filter is not None
            or preprocessor.contamination_checker is not None):
        raise ValueError("exact continuation cannot use generic record remapping or filtering")
    if not isinstance(review_policy, IndependentDatasetReviewPolicy):
        raise PermissionError("native independent dataset review policy required")
    review = review_policy.require_clearance(candidate, review_ref, preprocessor=preprocessor,
        maximum_bytes=preprocessor.config.max_input_bytes)
    if (review.source_file_hashes != {contract.source_file: contract.source_file_sha256}
            or review.tokenizer_ref != contract.tokenizer_ref
            or review.tokenizer_file_sha256 != contract.tokenizer_file_sha256
            or review.total_tokens != contract.eligible_tokens
            or review.unknown_tokens != contract.unknown_tokens):
        raise ValueError("independent clearance does not cover exact continuation inventory")
    acquisition = validate_remote_acquisition_for_preprocessing(candidate=candidate,
        selection=selection, approval=approval, receipt=receipt,
        max_bytes=preprocessor.config.max_input_bytes, relative_path=contract.source_file)
    if preprocessor.config.max_rows < contract.source_rows:
        raise ValueError("partial-source processing is prohibited for exact continuation")
    # Decode only the buffers whose exact SHA was authenticated, never a second open.
    original_tokenizer_path = checker.tokenizer_path.expanduser().resolve()
    records, report = reproduce_verified_bytes(acquisition.source_path.read_bytes(),
        original_tokenizer_path.read_bytes(), public_eval_pack_path.read_bytes(), contract)
    raw = b"".join(canonical_bytes(row) + b"\n" for row in records)
    output_sha = digest(raw)
    policy_sha = processing_profile_sha256(preprocessor)
    root = (preprocessor.config.artifact_root.expanduser().resolve()
            / "exact-continuation" / policy_sha[7:] / output_sha[7:])
    # Immutable content namespace includes lineage-independent bytes; per-run metadata
    # uses an actual run-ID hash so path traversal and slug collisions are impossible.
    artifact_dir = root / "runs" / sha256(run_id.encode("utf-8")).hexdigest()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    processed_path = root / "trainable-token-ids.jsonl"
    if processed_path.exists() and processed_path.read_bytes() != raw:
        raise ValueError("immutable continuation output path contains different bytes")
    if not processed_path.exists():
        # Publish a complete file atomically without replacing another writer.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".continuation-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, processed_path)
            except FileExistsError:
                if processed_path.read_bytes() != raw:
                    raise ValueError("concurrent immutable continuation output differs")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    evidence_path = artifact_dir / "processing_evidence.json"
    operations = ["exact_source_sha256", "original_tokenizer_sha256", "reviewed_row_commitment",
                  "within_row_token_id_windows", "prefix_label_masking", "no_eos_or_wrapper_insertion"]
    evidence = {
        "data_format": DATA_FORMAT, "continuation_contract": contract.to_dict(),
        "continuation_contract_ref": digest(canonical_bytes(contract.to_dict())),
        "original_tokenizer_path": str(original_tokenizer_path),
        "dataset_identity": f"{candidate.dataset_id}@{contract.revision}+{policy_sha}+{output_sha}",
        "candidate_id": candidate.candidate_id, "selection_decision_id": selection.decision_id,
        "approval_refs": list(acquisition.approval_refs), "source_content_hash": contract.source_file_sha256,
        "source_acquisition": {"kind": "remote_acquisition_receipt", "receipt_id": receipt.receipt_id},
        "independent_review_ref": review_ref, "processing_policy_hash": policy_sha,
        "processed_content_hash": output_sha, "processed_dataset_ref": str(processed_path),
        "processing_evidence_ref": str(evidence_path), "audit_scope": "full_scan",
        "filtering": {"kind": "exact_reviewed_selection", "selection_commitment_sha256": contract.selection_commitment_sha256},
        "deduplication": {"exact_status": "reviewed_normalized_duplicates_excluded"},
        "contamination": {"status": "independently_reviewed", "audit_refs": review.audit_refs,
                          "reference_manifest_refs": review.reference_manifest_refs},
        "preprocessing": {"operations": operations, "subset_screen": report},
        "chunking": {"kind": "within_row_token_ids", "prefix_tokens": contract.prefix_tokens,
                     "target_tokens": contract.target_tokens, "stride_tokens": contract.stride_tokens,
                     "output_records": len(records), "no_cross_row_stitching": True},
        "wrapper": {"kind": "no_text_wrapper", "template": f"token_ids:{contract.prefix_tokens}_prefix+{contract.target_tokens}_target"},
        "prompt_target_boundary": {"status": "explicit", "prefix_tokens": contract.prefix_tokens,
                                   "target_tokens": contract.target_tokens},
        "tokenizer_compatibility": {"status": "checked", "compatible": True, "tokenizer_ref": contract.tokenizer_ref,
                                   "tokenizer_file_sha256": contract.tokenizer_file_sha256},
    }
    manifest = build_dataset_manifest(run_id=run_id, lineage_id=lineage_id, stage_name=stage_name,
        candidate=candidate, revision=contract.revision, processed_dataset_ref=str(processed_path),
        processed_content_hash=output_sha, processed_bytes=len(raw), output_rows=len(records),
        mixture_weight=float(selection.mixture_weights.get(candidate.candidate_id, 1.0)),
        evidence=evidence, tokenizer_ref=contract.tokenizer_ref)
    prep, trainable = build_preprocessing_contracts(run_id=run_id, manifest_id=manifest.manifest_id,
        processed_dataset_ref=str(processed_path), operations=operations,
        dropped_examples=contract.source_rows-contract.accepted_rows, min_tokens=contract.prefix_tokens+contract.target_tokens)
    trainable.schema_version = DATA_FORMAT
    write_json(evidence_path, evidence)
    write_json(artifact_dir / "dataset_manifest.json", manifest.to_dict())
    write_json(artifact_dir / "preprocessing_report.json", prep.to_dict())
    write_json(artifact_dir / "trainable_data_contract.json", trainable.to_dict())
    return DataFactoryResult(candidate.candidate_id, evidence["dataset_identity"], output_sha,
        artifact_dir, manifest, prep, trainable, evidence)
