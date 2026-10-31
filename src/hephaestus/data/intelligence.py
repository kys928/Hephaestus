"""Bounded discovery/rejection/selection over real registered providers."""
import time
from dataclasses import dataclass
from hephaestus.schemas.discovery_contract import DatasetSearchRequest, DatasetSelectionDecision
from hephaestus.policy.dataset_review import candidate_sha256
from hephaestus.policy.internal_research import put_record
from hephaestus.storage.verified import read_verified_json
from hephaestus.utils.hashing import hash_json


def derive_data_requirements(diagnosis, *, stage, objective, tokenizer_ref, architecture,
                             providers, license_allowlist):
    targets = list(diagnosis.get("capability_gaps", []))
    if not targets or not stage or not objective or not tokenizer_ref or not architecture:
        raise ValueError("diagnosis, stage, objective, tokenizer and architecture required")
    natural = objective in {"pretraining", "causal_continuation", "natural_language_generalization"}
    return DatasetSearchRequest("data-requirements-" + hash_json({"diagnosis": diagnosis,
        "stage": stage, "objective": objective, "architecture": architecture})[:20],
        str(diagnosis["report_id"]), str(diagnosis["problem_statement"]),
        capability_targets=targets, required_languages=["en"], required_formats=["jsonl"],
        tokenizer_ref=tokenizer_ref, model_constraints=architecture,
        license_allowlist=list(license_allowlist), provider_allowlist=list(providers),
        metadata={"stage": stage, "objective": objective, "natural_language_required": natural})


@dataclass
class AutonomousDatasetIntelligence:
    registry: object
    inspector: object
    admission: object
    artifacts: object
    budget: object

    def discover_select(self, request, **scope):
        self.budget.validate()
        start, spent = time.monotonic(), 0
        discovered = self.registry.discover(request)
        discovery_ref = put_record(self.artifacts, {"schema": "autonomous-source-discovery.v1",
            "request": request.to_dict(), "candidates": [c.to_dict() for c in discovered.candidates],
            "issues": [i.to_dict() for i in discovered.issues]})
        refs, rejected, eligible, reviewed = [discovery_ref], {}, [], {}
        examined = 0
        for candidate in discovered.candidates:
            if (examined >= self.budget.maximum_candidates or spent >= self.budget.maximum_bytes or
                    time.monotonic() - start >= self.budget.maximum_seconds):
                rejected[candidate.candidate_id] = "discovery_budget_exhausted"
                continue
            examined += 1
            evidence = {"schema": "autonomous-candidate-decision.v1", "candidate": candidate.to_dict(),
                "candidate_sha256": candidate_sha256(candidate), "discovery_ref": discovery_ref}
            try:
                if request.metadata.get("natural_language_required") and candidate.metadata.get("synthetic"):
                    raise PermissionError("synthetic data cannot substitute for required natural-language corpus")
                review_ref, consumed = self.inspector.inspect(candidate, request,
                    maximum_bytes=self.budget.maximum_bytes - spent,
                    deadline=start + self.budget.maximum_seconds)
                if type(consumed) is not int or consumed < 0 or consumed > self.budget.maximum_bytes - spent:
                    raise ValueError("inspector exceeded source byte budget")
                spent += consumed
                if not set(candidate.task_types) & set(request.capability_targets):
                    raise PermissionError("candidate does not address diagnosed capability targets")
                if request.required_languages and not set(request.required_languages).issubset(candidate.languages):
                    raise PermissionError("candidate language evidence does not satisfy requirements")
                self.admission.require_clearance(candidate, review_ref, **scope)
                review = read_verified_json(self.artifacts, review_ref)
                scan = read_verified_json(self.artifacts, review["scan_ref"])
                # Explicit measured ranking. Eligibility always precedes score.
                relevance = len(set(candidate.task_types) & set(request.capability_targets)) / len(request.capability_targets)
                compatibility = 1.0 - scan["unknown_tokens"] / scan["total_tokens"]
                diversity = min(len(set(candidate.domains)) / 3.0, 1.0)
                components = {"relevance": relevance, "tokenizer_compatibility": compatibility,
                    "domain_diversity": diversity, "quality": 1.0, "provenance": 1.0,
                    "no_detected_internal_contamination": 1.0,
                    "expected_training_value_proxy": relevance * compatibility}
                score = sum(components.values()) / len(components)
                evidence.update(disposition="eligible", tier="A", review_ref=review_ref,
                                score=score, score_components=components)
                eligible.append((score, candidate, review_ref))
                reviewed[candidate.candidate_id] = review_ref
            except (ValueError, PermissionError, OSError, TypeError, KeyError) as exc:
                consumed = getattr(exc, "consumed", 0)
                if type(consumed) is int and 0 <= consumed <= self.budget.maximum_bytes - spent:
                    spent += consumed
                evidence["source_review_evidence_ref"] = getattr(exc, "evidence_ref", None)
                rejected[candidate.candidate_id] = type(exc).__name__ + ":" + str(exc)[:400]
                evidence.update(disposition="rejected", tier="B", reason=rejected[candidate.candidate_id])
            refs.append(put_record(self.artifacts, evidence))
        eligible.sort(key=lambda row: (-row[0], row[1].candidate_id))
        selected = eligible[:1]
        decision = DatasetSelectionDecision("", request.request_id,
            "selected" if selected else "inconclusive",
            selected_candidate_ids=[r[1].candidate_id for r in selected],
            ranked_candidate_ids=[r[1].candidate_id for r in eligible], rejected_candidates=rejected,
            selection_rationale="Measured eligibility precedes explicit ranking; rejected candidates do not stop search.",
            required_approvals=["dataset_acquisition", "dataset_selection_approval"] if selected else [],
            evidence_refs=refs, metadata={"claim_scope": "internal_prospective", "tier": "A" if selected else None,
                "policy_ref": self.admission.policy_ref, "candidate_review_refs": reviewed,
                "examined_candidates": examined, "source_bytes": spent,
                "external_certification": False, "budget": self.budget.to_dict()})
        decision.decision_id = "internal-data-selection-" + hash_json(decision.to_dict())[:24]
        return decision, put_record(self.artifacts, decision.to_dict())
