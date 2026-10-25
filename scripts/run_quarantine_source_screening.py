#!/usr/bin/env python3
"""Exact train-file quarantine scans. No training or protected reference access.

Source bytes live only in TemporaryDirectory, never in uploaded evidence.
Only aggregate measurements, identities and pending custody requests persist.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import platform
from importlib.metadata import version
from urllib.request import Request, urlopen

from hephaestus.data.quarantine_screening import (
    normalized, pii_counts, duplicate_counts, public_probe_overlap,
)
from hephaestus.data.normalization import normalize_text
from hephaestus.state.screening_store import ScreeningStore
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "configs/data/quarantine_source_screening_v1.json"


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def verify_scope(scope, accepted_scope_ref):
    if accepted_scope_ref != digest(canonical_bytes(scope)):
        raise PermissionError("exact operator-accepted source scope digest required")
    if (scope.get("schema") != "quarantine-source-screening-scope.v1"
            or any(scope.get(k) is not False for k in (
                "training_allowed", "redistribution_allowed", "protected_reference_access_allowed",
                "candidate_selection_approved"))):
        raise PermissionError("scope must prohibit training, redistribution and private reference access")
    if len(scope.get("sources", [])) != 2 or not scope.get("operator_authorization_basis"):
        raise PermissionError("two explicitly reviewed source files required")


def fetch_bounded(url, maximum_bytes):
    with urlopen(Request(url, headers={"User-Agent": "hephaestus-quarantine-audit/1"}), timeout=60) as response:
        data = response.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise ValueError("screening source exceeds authorized byte ceiling")
    return data


def public_prompts(pack):
    return [p["prompt"] for field in ("generation_probes", "continuation_prompts",
        "structure_tests", "repetition_checks", "length_termination_checks", "regression_prompts")
        for p in pack.get(field, []) if isinstance(p.get("prompt"), str)]


def mapped_profile_text(row, *, is_saf):
    """Match the existing preview's field normalization and newline boundary.

    Flattening newlines is a distinct study, never a drop-in eligibility repair.
    """
    if is_saf:
        return normalize_text(row["question"]) + "\n" + normalize_text(row["reference_answer"])
    return normalize_text(row["text"])


def token_measurement(tokenizer, texts, unknown_id):
    total = unknown = 0
    for text in texts:
        ids = tokenizer.encode(text).ids
        total += len(ids)
        unknown += ids.count(unknown_id)
    return {"total_tokens": total, "unknown_tokens": unknown,
            "maximum_unknown_fraction": .01,
            "compatible": total > 0 and unknown * 100 <= total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-scope-ref", required=True)
    parser.add_argument("--output", type=Path, default=Path("quarantine_source_screening"))
    args = parser.parse_args()
    scope = json.loads(SCOPE.read_text())
    verify_scope(scope, args.accepted_scope_ref)
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = FileSystemArtifactStore(args.output / "artifacts")
    state = ScreeningStore(args.output / "state")
    def put(value):
        ref = artifacts.put_bytes(canonical_bytes(value), media_type="application/json").artifact_ref
        state.append({"kind": value.get("schema", "quarantine-measurement"), "artifact_ref": ref})
        return ref
    scope_ref = put(scope)
    # Public pack only; preserve exact original bytes without modifying evals.
    pack_path = ROOT / "configs/eval_packs/semantic_behavior_v1.yaml"
    pack_bytes = pack_path.read_bytes()
    prompts = public_prompts(json.loads(pack_bytes))
    pack_ref = digest(pack_bytes)
    # Do not read any unknown/private text in the search for a custodian.
    custody = {"schema": "custodial-reference-provisioning-request.v1",
        "scope_ref": scope_ref, "status": "external_custodian_required",
        "scope_ref_is_source_measurement_only_not_protected_audit_permission": True,
        "separately_accepted_protected_audit_scope_ref": None,
        "reference_manifest_refs": {}, "trusted_custodian_id": None,
        "trusted_custodian_public_key": None,
        "required_roles": ["training", "development", "diagnostic", "holdout"],
        "execution_boundary": "custodian_environment",
        "public_probe_pack_file_sha256": pack_ref,
        "public_pack_is_not_a_sealed_holdout": True,
        "known_public_reference_metadata_not_complete_custodial_manifests": {
            "original_training_bytes_ref": "sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c",
            "original_training_location": "hephaestus/scientific/v1/runtime_bindings/first-bounded-scientific-training-001-33866198758/dataset/trainable.jsonl",
            "diagnostic_probe_repository_file": "docs/evidence/first-targeted-post-failure-diagnostics-001-33874832481/probe.json",
            "diagnostic_probe_file_sha256": "sha256:c943334f2f6e7162380be74748506d26dd13542fece21d42672d35f3abbfc2e9",
            "development_public_probe_pack_file_sha256": pack_ref,
            "independent_holdout": None,
            "these_are_incomplete_exposure_metadata_not_accepted_reference_manifests": True},
        "required_inputs": ["four complete original-text inventory manifests",
            "independently pinned Ed25519 public key", "signed exact-scope read authorization",
            "custodian-side lexical and semantic audits; only aggregate receipts returned"],
        "training_allowed": False}
    custody_ref = put(custody)
    # Real original tokenizer is loaded from read-only storage in CI, never replaced.
    from run_first_autonomous_dataset_discovery import s3_client, required, get_s3_bytes, TOKENIZER_KEY, TOKENIZER_ID
    from run_schema_aware_public_preview_review import EXPECTED_TOKENIZER_BYTES_SHA256, english_likelihood, window_proof
    from tokenizers import Tokenizer
    raw = get_s3_bytes(s3_client(), required("RUNPOD_NETWORK_VOLUME_ID"), TOKENIZER_KEY)
    if hashlib.sha256(raw).hexdigest() != EXPECTED_TOKENIZER_BYTES_SHA256:
        raise ValueError("original tokenizer bytes mismatch")
    tokenizer = Tokenizer.from_str(raw.decode())
    unknown_id = tokenizer.token_to_id("<|unk|>")
    if unknown_id is None:
        raise ValueError("original unknown token missing")
    reports = []
    import pyarrow.parquet as pq
    with tempfile.TemporaryDirectory(prefix="hephaestus-source-quarantine-") as temporary:
        for index, spec in enumerate(scope["sources"]):
            url = f"https://huggingface.co/datasets/{spec['dataset_id']}/resolve/{spec['revision']}/{spec['file']}"
            # Rights/source-card identity bound to the SAME repository revision.
            card_url = f"https://huggingface.co/datasets/{spec['dataset_id']}/resolve/{spec['revision']}/README.md"
            card = fetch_bounded(card_url, 128 * 1024)
            expected_license = "cc-by-4.0" if "Feedback" in spec["dataset_id"] else "cc-by-sa-3.0"
            if expected_license not in card.decode().lower():
                raise ValueError("source rights declaration differs from reviewed scope")
            rights_ref = put({"schema": "source-rights-screening-review.v1", "dataset_id": spec["dataset_id"],
                "revision": spec["revision"], "card_url": card_url, "card_sha256": digest(card),
                "basis": spec["rights_basis"], "decision": spec["rights_status"],
                "training_rights_cleared": False, "legal_certification": False})
            data = fetch_bounded(url, spec["bytes"])
            if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError("source file bytes differ from immutable reviewed inventory")
            local = Path(temporary) / f"source-{index}.parquet"
            local.write_bytes(data)
            file = pq.ParquetFile(local)
            if file.metadata.num_rows != spec["rows"]:
                raise ValueError("source row count differs from reviewed inventory")
            rows = file.read().to_pylist()
            is_saf = "Feedback" in spec["dataset_id"]
            fields = ["question", "reference_answer"] if is_saf else ["text"]
            if any(not all(isinstance(row.get(k), str) for k in fields) for row in rows):
                raise ValueError("source schema does not match reviewed mapping")
            texts = [" ".join(row[k] for k in fields) for row in rows]
            if sum(map(len, texts)) > 100_000_000:
                raise ValueError("decoded source exceeds audit resource ceiling")
            accepted, flattened, rejected = [], [], Counter()
            preview_texts = []
            for row, text in zip(rows, texts, strict=True):
                keep, reason = english_likelihood(text)
                if keep:
                    accepted.append(mapped_profile_text(row, is_saf=is_saf))
                    flattened.append(normalized(text))
                else:
                    rejected[reason] += 1
            for row in rows[:100]:
                text = mapped_profile_text(row, is_saf=is_saf)
                if english_likelihood(text)[0]:
                    preview_texts.append(text[:24000])
            report = {"schema": "quarantined-source-screening-report.v1", "source": spec,
                "scope_ref": scope_ref, "rights_ref": rights_ref,
                "source_file_sha256": digest(data), "source_rows_checked": len(rows),
                "source_coverage": "all_rows_of_exact_named_train_file",
                "pii_triage": pii_counts(rows), "mapped_field_duplicates": duplicate_counts(texts),
                "public_probe_overlap": public_probe_overlap(texts, prompts),
                "public_probe_pack_file_sha256": pack_ref,
                "normalization": "existing normalize_text per field; preserve line breaks and prompt-target newline; no vocabulary replacement",
                "english_heuristic_kept_rows": len(accepted), "english_heuristic_rejected": dict(rejected),
                "tokenizer": {"tokenizer_ref": TOKENIZER_ID,
                    "file_sha256": EXPECTED_TOKENIZER_BYTES_SHA256,
                    **token_measurement(tokenizer, accepted, unknown_id)},
                "first_100_existing_preview_profile_recheck": token_measurement(tokenizer, preview_texts, unknown_id),
                "flattened_whitespace_study": {
                    "normalization": "NFKC plus all whitespace flattened to spaces; distinct unapproved profile",
                    "changes_line_boundaries": True, "eligibility_change_authorized": False,
                    **token_measurement(tokenizer, flattened, unknown_id)},
                "tokenizer_control_counts": {label: token_measurement(tokenizer, [text], unknown_id)
                    for label, text in {"newline": "\n", "space": " ", "line_preserving": "one\ntwo",
                        "space_separated": "one two"}.items()},
                "semantic_contamination": {"status": "not_run", "reason": "custodial originals unavailable"},
                "protected_reference_leakage": {"status": "not_run", "reference_manifest_refs": {}},
                "custody_request_ref": custody_ref, "training_allowed": False,
                "dataset_approved": False, "independent_review_accepted": False,
                "certification_state": "not_certified"}
            if not is_saf:
                proof = window_proof(tokenizer, accepted)
                proof["train_preview_only"] = False
                proof["source_scope"] = "all_English_heuristic_accepted_train_rows"
                report["continuation_window_proof"] = proof
            report_ref = put(report)
            reports.append({"dataset_id": spec["dataset_id"], "report_ref": report_ref, "report": report})
            local.unlink()
    if pack_path.read_bytes() != pack_bytes:
        raise ValueError("frozen public pack changed during screening")
    summary = {"schema": "source-screening-campaign.v1", "source_sha": os.environ.get("GITHUB_SHA"),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            "tokenizers": version("tokenizers"), "pyarrow": version("pyarrow"),
            "cryptography": version("cryptography")},
        "scope_ref": scope_ref, "reports": reports, "custody_request_ref": custody_ref,
        "all_protected_audits_complete": False, "source_quarantine_deleted": True,
        "existing_cpu_campaign_progress": "11/45 unchanged", "training_executed": False,
        "dataset_approved": False, "certification_state": "not_certified"}
    put(summary)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sources_scanned": len(reports), "protected_audits_complete": False,
        "source_rows": [r["report"]["source_rows_checked"] for r in reports]}))


if __name__ == "__main__":
    main()
