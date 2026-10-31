"""Bind source review evidence to a deterministic selection proposal."""
from __future__ import annotations

from dataclasses import dataclass

from hephaestus.policy.dataset_review import candidate_sha256, preview_issues
from hephaestus.utils.hashing import hash_json


@dataclass(slots=True)
class ReviewBoundDatasetSelector:
    delegate: object
    review_policy: object
    review_refs: dict[str, str]  # exact candidate identity, never provider assertions

    def select(self, request, candidates):
        eligible, rejected, refs = [], {}, {}
        for candidate in candidates:
            if candidate.provider_id != "huggingface":
                rejected[candidate.candidate_id] = "reviewed_huggingface_source_required"
                continue
            ref = self.review_refs.get(candidate.candidate_id)
            if not ref:
                rejected[candidate.candidate_id] = "dataset_review_missing"
                continue
            try:
                review = self.review_policy.review(ref)
                issues = preview_issues(review, candidate,
                    tokenizer_ref=self.review_policy.tokenizer_ref,
                    tokenizer_sha256=self.review_policy.tokenizer_sha256)
                refs[candidate.candidate_id] = ref
                if issues:
                    rejected[candidate.candidate_id] = ";".join(issues)
                else:
                    eligible.append(candidate)
            except (ValueError, OSError, TypeError, KeyError):
                rejected[candidate.candidate_id] = "dataset_review_invalid"
        decision = self.delegate.select(request, eligible)
        decision.rejected_candidates.update(rejected)
        decision.evidence_refs = sorted(set([*decision.evidence_refs, *refs.values()]))
        decision.metadata["candidate_review_refs"] = refs
        if decision.selected_candidate_ids:
            if len(decision.selected_candidate_ids) != 1:
                raise ValueError("review-bound remote handoff requires exactly one selected source")
            candidate = next(c for c in eligible if c.candidate_id == decision.selected_candidate_ids[0])
            ref = refs[candidate.candidate_id]
            decision.metadata.update(dataset_review_ref=ref,
                candidate_sha256=candidate_sha256(candidate))
            decision.required_approvals = sorted(set([
                *decision.required_approvals, "dataset_selection_approval", "dataset_acquisition",
                "dataset_review:" + ref,
            ]))
        # Changed source, review, or rejection evidence MUST invalidate any prior approval.
        decision.decision_id = "reviewed-dataset-selection-" + hash_json(decision.to_dict())[:24]
        return decision
