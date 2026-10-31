"""Immutable scientific-state contrast assessment; NOT an approval or a certificate.

Scores real immutable prediction artifacts against an externally pinned holdout.
This module intentionally cannot declare a model certified or activate a role.
Those actions remain under existing scientific/promotion/approval policies.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from hephaestus.storage.base import ArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json


@dataclass(frozen=True, slots=True)
class ScientificStateAssessment:
    artifact_ref: str
    eligible_for_review: bool
    blocking_issues: tuple[str, ...]
    candidate_accuracy: float
    baseline_accuracy: float
    candidate_contrast_accuracy: float
    protected_regressions: int


def _case_index(pack: dict) -> tuple[dict[str, dict], dict[str, list[str]]]:
    if pack.get("schema") != "scientific-state-holdout.v1" or pack.get("split") != "sealed_holdout":
        raise ValueError("expected a sealed scientific-state holdout manifest")
    cases = pack.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("empty or invalid holdout cases")
    indexed: dict[str, dict] = {}
    contrasts: dict[str, list[str]] = defaultdict(list)
    states: set[str] = set()
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("malformed holdout case")
        case_id, gold, pair = item.get("case_id"), item.get("gold_state"), item.get("contrast_id")
        if not all(isinstance(v, str) and v and len(v) <= 180 for v in (case_id, gold, pair)):
            raise ValueError("case has missing/invalid identities")
        if case_id in indexed or type(item.get("protected")) is not bool:
            raise ValueError("duplicate case or missing protected label")
        indexed[case_id] = item
        contrasts[pair].append(case_id)
        states.add(gold)
    if len(states) < 2 or any(len(members) != 2 for members in contrasts.values()):
        raise ValueError("holdout requires at least two states and exact contrast pairs")
    if not any(item["protected"] for item in indexed.values()):
        raise ValueError("holdout requires protected regression cases")
    return indexed, contrasts


def _predictions(payload: dict, *, pack_ref: str, case_ids: set[str], seed: int,
                 settings_ref: str) -> tuple[str, dict[str, str]]:
    if (payload.get("schema") != "scientific-state-predictions.v1"
            or payload.get("pack_ref") != pack_ref
            or payload.get("seed") != seed
            or payload.get("settings_ref") != settings_ref):
        raise ValueError("prediction pack/seed/generation-settings mismatch")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"]:
        raise ValueError("prediction run id missing")
    checkpoint = payload.get("checkpoint_ref")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("prediction checkpoint identity missing")
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise ValueError("prediction rows missing")
    output: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"case_id", "state"}:
            raise ValueError("prediction entry must contain exactly case_id and state")
        case_id, state = row["case_id"], row["state"]
        if not isinstance(case_id, str) or not isinstance(state, str) or not state:
            raise ValueError("malformed prediction")
        if case_id in output:
            raise ValueError("duplicate prediction for one case")
        output[case_id] = state
    if set(output) != case_ids:
        raise ValueError("predictions do not exactly cover the sealed holdout")
    return checkpoint, output


def assess_scientific_states(
    artifacts: ArtifactStore, *, approved_pack_ref: str,
    baseline_refs: Sequence[str], candidate_refs: Sequence[str],
    diagnostic_manifest_ref: str | None,
    minimum_accuracy: float, minimum_contrast_accuracy: float,
    minimum_per_state_accuracy: float = 0.5, minimum_matched_seeds: int = 2,
) -> ScientificStateAssessment:
    """Recompute metrics from immutable case/prediction bytes, no ML inference.

    The approved pack ref MUST originate from an external, governed policy
    rather than being synthesized from a candidate-controlled artifact. A
    diagnostic source manifest is required to check known overlap, but even a
    disjoint manifest cannot independently prove genuinely unseen holdouts.
    """
    for threshold in (minimum_accuracy, minimum_contrast_accuracy, minimum_per_state_accuracy):
        if type(threshold) not in (float, int) or not 0 <= threshold <= 1:
            raise ValueError("scientific-state thresholds must be finite fractions")
    if minimum_matched_seeds < 2 or len(baseline_refs) < minimum_matched_seeds:
        raise ValueError("insufficient matched repeated scientific-state runs")
    if len(baseline_refs) != len(candidate_refs) or len(set(baseline_refs)) != len(baseline_refs):
        raise ValueError("baseline/candidate repetition identities are malformed")
    if len(set(candidate_refs)) != len(candidate_refs):
        raise ValueError("candidate repeated artifacts must be distinct")
    pack = read_verified_json(artifacts, approved_pack_ref)
    cases, contrasts = _case_index(pack)
    case_ids = set(cases)
    reasons: list[str] = []
    if not diagnostic_manifest_ref:
        reasons.append("diagnostic_overlap_not_verified")
    else:
        diagnostics = read_verified_json(artifacts, diagnostic_manifest_ref)
        if diagnostics.get("schema") != "scientific-diagnostic-cases.v1":
            raise ValueError("diagnostic case manifest schema mismatch")
        rows = diagnostics.get("case_ids")
        if not isinstance(rows, list) or not all(isinstance(x, str) for x in rows):
            raise ValueError("diagnostic case identities missing")
        if case_ids.intersection(rows):
            reasons.append("known_holdout_contamination")

    runs = []
    baseline_checkpoint: str | None = None
    candidate_checkpoint: str | None = None
    settings_ref: str | None = None
    seeds: set[int] = set()
    for bre, cre in zip(baseline_refs, candidate_refs):
        b = read_verified_json(artifacts, bre)
        c = read_verified_json(artifacts, cre)
        seed = b.get("seed")
        if type(seed) is not int or seed < 0 or seed in seeds or c.get("seed") != seed:
            raise ValueError("repeated runs require distinct matched nonnegative seeds")
        seeds.add(seed)
        settings = b.get("settings_ref")
        if not isinstance(settings, str) or not settings.startswith("sha256:") or c.get("settings_ref") != settings:
            raise ValueError("repeated generation settings must match")
        # Verify the actual referenced decoding/settings artifact, not just its name.
        read_verified_json(artifacts, settings)
        if settings_ref is None:
            settings_ref = settings
        elif settings_ref != settings:
            raise ValueError("different generation settings across repeated runs")
        bcheckpoint, bp = _predictions(b, pack_ref=approved_pack_ref, case_ids=case_ids,
                                       seed=seed, settings_ref=settings)
        ccheckpoint, cp = _predictions(c, pack_ref=approved_pack_ref, case_ids=case_ids,
                                       seed=seed, settings_ref=settings)
        if baseline_checkpoint is None:
            baseline_checkpoint, candidate_checkpoint = bcheckpoint, ccheckpoint
        elif baseline_checkpoint != bcheckpoint or candidate_checkpoint != ccheckpoint:
            raise ValueError("checkpoint changed between repeated trials")
        if bcheckpoint == ccheckpoint:
            raise ValueError("baseline and candidate use the same checkpoint")
        runs.append((seed, bp, cp))

    correct_b = correct_c = paired_c = protected_regressions = 0
    per_state: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for _, b, c in runs:
        for case_id, spec in cases.items():
            gold = spec["gold_state"]
            b_ok, c_ok = b[case_id] == gold, c[case_id] == gold
            correct_b += b_ok
            correct_c += c_ok
            per_state[gold][0] += c_ok
            per_state[gold][1] += 1
            if spec["protected"] and b_ok and not c_ok:
                protected_regressions += 1
        for member_ids in contrasts.values():
            paired_c += all(c[case_id] == cases[case_id]["gold_state"] for case_id in member_ids)
    n = len(runs) * len(cases)
    baseline_accuracy = correct_b / n
    candidate_accuracy = correct_c / n
    contrast_accuracy = paired_c / (len(runs) * len(contrasts))
    per_state_accuracy = {key: good / total for key, (good, total) in sorted(per_state.items())}

    if candidate_accuracy <= baseline_accuracy:
        reasons.append("no_measured_improvement")
    if candidate_accuracy < minimum_accuracy:
        reasons.append("minimum_accuracy_not_met")
    if contrast_accuracy < minimum_contrast_accuracy:
        reasons.append("contrast_accuracy_not_met")
    if min(per_state_accuracy.values()) < minimum_per_state_accuracy:
        reasons.append("protected_state_coverage_not_met")
    if protected_regressions:
        reasons.append("protected_regression_detected")
    report = {
        "schema": "scientific-state-assessment.v1",
        "holdout_ref": approved_pack_ref, "diagnostic_manifest_ref": diagnostic_manifest_ref,
        "baseline_refs": list(baseline_refs), "candidate_refs": list(candidate_refs),
        "baseline_checkpoint_ref": baseline_checkpoint,
        "candidate_checkpoint_ref": candidate_checkpoint,
        "generation_settings_ref": settings_ref, "matched_seeds": sorted(seeds),
        "metrics": {
            "baseline_accuracy": baseline_accuracy, "candidate_accuracy": candidate_accuracy,
            "candidate_contrast_accuracy": contrast_accuracy,
            "per_state_accuracy": per_state_accuracy, "protected_regressions": protected_regressions,
        },
        "criteria": {
            "minimum_accuracy": minimum_accuracy,
            "minimum_contrast_accuracy": minimum_contrast_accuracy,
            "minimum_per_state_accuracy": minimum_per_state_accuracy,
            "minimum_matched_seeds": minimum_matched_seeds,
        },
        "blocking_issues": reasons,
        "eligible_for_external_certification_review": not reasons,
        "certification_state": "not_certified",
        "holdout_unseen_provenance_independently_verified": False,
    }
    artifact = artifacts.put_bytes(canonical_bytes(report), media_type="application/json")
    return ScientificStateAssessment(
        artifact.artifact_ref, not reasons, tuple(reasons), candidate_accuracy,
        baseline_accuracy, contrast_accuracy, protected_regressions,
    )
