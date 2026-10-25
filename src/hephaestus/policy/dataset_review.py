"""Fail-closed scientific eligibility, separate from human action approval.

Trusted review/scope digests are injected by the independent operator, never
read from candidate metadata. Hashes attest bytes, not scientific correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from hephaestus.schemas.dataset_review import DatasetSourceReview, DatasetScreeningAccessScope
from hephaestus.schemas.discovery_contract import DatasetCandidate
from hephaestus.storage.verified import read_verified_json
from hephaestus.utils.hashing import hash_json


AUDITS = frozenset({"pii", "exact_near_duplicate", "semantic_contamination", "frozen_evaluation_leakage"})
REFERENCES = frozenset({"training", "development", "diagnostic", "holdout"})


def candidate_sha256(candidate: DatasetCandidate) -> str:
    """Bind the source and file scope, independent of ranking annotations."""
    return "sha256:" + hash_json({
        "candidate_id": candidate.candidate_id, "provider_id": candidate.provider_id,
        "dataset_id": candidate.dataset_id, "revision": candidate.revision,
        "license": candidate.license,
        "acquisition_files": sorted(candidate.metadata.get("acquisition_files", [])),
    })


def processing_profile_sha256(preprocessor) -> str:
    cfg = preprocessor.config
    return "sha256:" + hash_json({
        "max_rows": cfg.max_rows, "chunk_size_tokens": cfg.chunk_size_tokens,
        "min_tokens": cfg.min_tokens, "near_duplicate_threshold": cfg.near_duplicate_threshold,
        "prompt_target_template": cfg.prompt_target_template,
        "filter_id": getattr(preprocessor.record_filter, "filter_id", None),
        "contamination_reference": getattr(preprocessor.contamination_checker, "reference_set_id", None),
        "tokenizer_ref": getattr(preprocessor.tokenizer_checker, "tokenizer_ref", None),
        "tokenizer_checker_id": getattr(preprocessor.tokenizer_checker, "checker_id", None),
        "record_mapper_id": getattr(preprocessor.record_mapper, "mapper_id", None),
    })


def preview_issues(review: DatasetSourceReview, candidate: DatasetCandidate, *,
                   tokenizer_ref: str, tokenizer_sha256: str) -> list[str]:
    issues = []
    if review.schema != "dataset-source-review.v1" or review.scope not in {"public_preview", "full_corpus"}:
        issues.append("invalid_dataset_review_schema")
    if review.candidate_sha256 != candidate_sha256(candidate):
        issues.append("dataset_review_source_mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", candidate.revision or "") or review.revision_verified is not True:
        issues.append("dataset_preview_revision_unverified")
    for field in ("schema_status", "language_status", "capability_status"):
        if getattr(review, field) != "pass":
            issues.append("dataset_review_" + field)
    if review.rights_status != "cleared" or not review.rights_evidence_refs:
        issues.append("dataset_source_rights_unresolved")
    if (review.tokenizer_ref != tokenizer_ref or
            review.tokenizer_file_sha256 != tokenizer_sha256):
        issues.append("dataset_review_tokenizer_mismatch")
    # Strict integer counts. No NaN, rounded-rate, boolean or changed-threshold bypass.
    if (type(review.total_tokens) is not int or type(review.unknown_tokens) is not int or
            review.total_tokens <= 0 or not 0 <= review.unknown_tokens <= review.total_tokens or
            review.unknown_tokens * 100 > review.total_tokens):
        issues.append("dataset_unknown_token_limit_exceeded")
    if not review.normalization_profile or not review.reviewer_id:
        issues.append("dataset_review_missing_method_or_reviewer")
    return issues


@dataclass(frozen=True, slots=True)
class IndependentDatasetReviewPolicy:
    artifacts: object
    tokenizer_ref: str
    tokenizer_sha256: str
    trusted_review_refs: frozenset[str] = frozenset()
    trusted_access_scope_refs: frozenset[str] = frozenset()

    def review(self, ref: str) -> DatasetSourceReview:
        return DatasetSourceReview(**read_verified_json(self.artifacts, ref))

    def require_clearance(self, candidate: DatasetCandidate, ref: str, *, preprocessor,
                          maximum_bytes: int) -> DatasetSourceReview:
        if ref not in self.trusted_review_refs:
            raise PermissionError("independent dataset review digest has not been accepted")
        review = self.review(ref)
        issues = preview_issues(review, candidate, tokenizer_ref=self.tokenizer_ref,
                                tokenizer_sha256=self.tokenizer_sha256)
        if issues:
            raise ValueError(";".join(issues))
        files = candidate.metadata.get("acquisition_files")
        if (not isinstance(files, list) or not files or len(files) != len(set(files)) or
                set(files) != set(review.source_file_hashes)):
            raise ValueError("full-corpus review must cover exactly the selected files")
        if review.scope != "full_corpus" or set(review.audit_refs) != AUDITS:
            raise ValueError("full-corpus PII, duplicate, semantic and frozen-eval audits required")
        if set(review.reference_manifest_refs) != REFERENCES:
            raise ValueError("all four protected reference manifests are required")
        if review.access_scope_ref not in self.trusted_access_scope_refs:
            raise PermissionError("separately authorized screening access is required")
        access = DatasetScreeningAccessScope(**read_verified_json(self.artifacts, review.access_scope_ref))
        if (access.schema != "dataset-screening-access-scope.v1" or
                access.status != "authorized" or not access.authorized_by or
                access.training_allowed is not False or
                access.candidate_sha256 != candidate_sha256(candidate) or
                set(access.acquisition_files) != set(files) or
                type(access.maximum_bytes) is not int or
                not 0 < access.maximum_bytes <= maximum_bytes or
                not AUDITS.issubset(access.purposes) or
                access.reference_manifest_refs != review.reference_manifest_refs):
            raise PermissionError("screening scope does not cover this source and its protected references")
        if (preprocessor.tokenizer_checker is None or
                preprocessor.tokenizer_checker.tokenizer_ref != self.tokenizer_ref or
                review.processing_profile_sha256 != processing_profile_sha256(preprocessor)):
            raise ValueError("frozen tokenizer checker and exact preprocessing profile are required")
        for evidence_ref in [*review.rights_evidence_refs, *review.reference_manifest_refs.values()]:
            read_verified_json(self.artifacts, evidence_ref)
        for digest in review.source_file_hashes.values():
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError("source file must have an audited sha256")
        for name, evidence_ref in review.audit_refs.items():
            report = read_verified_json(self.artifacts, evidence_ref)
            if (report.get("audit") != name or report.get("status") != "pass" or
                    report.get("coverage") != "full_corpus" or not report.get("method") or
                    report.get("candidate_sha256") != review.candidate_sha256 or
                    report.get("source_file_hashes") != review.source_file_hashes or
                    report.get("reference_manifest_refs") != review.reference_manifest_refs or
                    report.get("access_scope_ref") != review.access_scope_ref):
                raise ValueError("audit scope, method or outcome differs from accepted review")
        return review
