#!/usr/bin/env python3
"""Real pinned WikiText-2 TRAIN-file screening for one compatible subset.

Read-only S3 original tokenizer and HF source access. Raw text stays in
temporary quarantine, deleted on exit. No training inputs are published.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from hephaestus.data.continuation_eligibility import screen_continuation_rows
from hephaestus.storage.verified import canonical_bytes

from run_quarantine_source_screening import (
    SCOPE, ROOT, verify_scope, fetch_bounded, public_prompts,
)
from run_schema_aware_public_preview_review import (
    EXPECTED_TOKENIZER_BYTES_SHA256, english_likelihood,
)
from run_first_autonomous_dataset_discovery import (
    s3_client, required, get_s3_bytes, TOKENIZER_KEY, TOKENIZER_ID,
)

# This pre-existing source-only screening scope contains the exact chosen file.
ACCEPTED_SCOPE_REF = "sha256:d7b3e42bad911f14095fb4b1bd9d0caa434a7eea7bec52472b70effcdffced69"
REPORT = Path("wikitext_continuation_subset_quarantine.json")


def produce_report() -> dict[str, object]:
    from tokenizers import Tokenizer
    import pyarrow.parquet as pq

    scope = json.loads(SCOPE.read_text(encoding="utf-8"))
    verify_scope(scope, ACCEPTED_SCOPE_REF)
    matched = [s for s in scope["sources"] if s["dataset_id"] == "Salesforce/wikitext"]
    if len(matched) != 1:
        raise ValueError("single source identity not unique in accepted scope")
    source = matched[0]
    if source["rights_status"] != "screening_only_no_training_clearance":
        raise PermissionError("source authorization scope changed")
    original = get_s3_bytes(s3_client(), required("RUNPOD_NETWORK_VOLUME_ID"), TOKENIZER_KEY)
    if sha256(original).hexdigest() != EXPECTED_TOKENIZER_BYTES_SHA256:
        raise ValueError("original frozen tokenizer bytes differ from expected SHA")
    tokenizer = Tokenizer.from_str(original.decode("utf-8"))
    unknown_id = tokenizer.token_to_id("<|unk|>")
    if unknown_id is None:
        raise ValueError("frozen tokenizer unknown ID unavailable")

    base = (f"https://huggingface.co/datasets/{source['dataset_id']}/"
            f"resolve/{source['revision']}/")
    card = fetch_bounded(base + "README.md", 128 * 1024)
    card_text = card.decode("utf-8").lower()
    if not all(label in card_text for label in ("cc-by-sa-3.0", "cc-by-sa-4.0")):
        raise ValueError("unresolved license declarations changed; halt for review")
    file_path = ROOT / "configs/eval_packs/semantic_behavior_v1.yaml"
    pack = file_path.read_bytes()
    prompts = tuple(public_prompts(json.loads(pack)))
    with tempfile.TemporaryDirectory(prefix="hephaestus-wikitext-source-only-") as temporary:
        raw = fetch_bounded(base + source["file"], source["bytes"])
        if (len(raw) != source["bytes"]
                or sha256(raw).hexdigest() != source["sha256"]):
            raise ValueError("pinned source bytes differ from scope")
        local = Path(temporary) / "source.parquet"
        local.write_bytes(raw)
        parquet = pq.ParquetFile(local)
        if parquet.metadata.num_rows != source["rows"]:
            raise ValueError("pinned row count differs from scope")
        rows = parquet.read().to_pylist()
        measured = screen_continuation_rows(
            rows, tokenizer, unknown_id,
            language_check=lambda t: bool(english_likelihood(t)[0]),
            public_prompts=prompts,
        )
    if file_path.read_bytes() != pack:
        raise ValueError("frozen public evaluation pack mutated")
    return {
        **measured,
        "scope_ref": ACCEPTED_SCOPE_REF,
        "dataset_id": source["dataset_id"],
        "revision": source["revision"],
        "source_file": source["file"],
        "source_file_sha256": "sha256:" + source["sha256"],
        "card_sha256": "sha256:" + sha256(card).hexdigest(),
        "rights": {
            "source_declarations": ["cc-by-sa-3.0", "gfdl", "cc-by-sa-4.0"],
            "status": "unresolved_attribution_provenance_and_license_obligations",
            "not_training_cleared": True,
        },
        "tokenizer_ref": TOKENIZER_ID,
        "tokenizer_file_sha256": EXPECTED_TOKENIZER_BYTES_SHA256,
        "public_eval_pack_sha256": "sha256:" + sha256(pack).hexdigest(),
        "scientific_capability": "english_causal_lm_continuation_only_not_json_or_short_answers",
        "reference_manifest_refs": {},
        "full_source_privacy_clearance": False,
        "semantic_contamination_audit": "not_run_no_external_custodian",
        "frozen_evaluation_leakage_audit": "not_run_no_external_custodian",
        "selection_or_acquisition_approval_ref": None,
        "scientific_verdict": "HOLD_pending_independent_rights_and_reference_custody",
    }


def main() -> None:
    result = produce_report()
    if result["accepted_unique_rows"] < 1 or result["eligible_within_row_windows"] < 1:
        raise ValueError("candidate has no actual qualifying continuation training utility")
    REPORT.write_bytes(canonical_bytes(result))
    print(json.dumps({
        "source": result["dataset_id"], "rows": result["source_rows_examined"],
        "accepted": result["accepted_unique_rows"],
        "windows": result["eligible_within_row_windows"],
        "unknown_tokens": result["unknown_tokens"],
        "eligible_tokens": result["eligible_tokens"],
        "clearance": result["scientific_verdict"],
    }))


if __name__ == "__main__":
    main()
