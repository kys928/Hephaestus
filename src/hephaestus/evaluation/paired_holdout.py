"""Issue #152: strict paired, precommitted held-out JSON assessment.

No training, downloads, raw-answer publication, model promotion or certificates.
Caller must independently custody holdout and generation receipts.
"""
from __future__ import annotations

from .exact_json import SCORER_VERSION, strict_json_object, exact_typed_equal, score_prediction

from collections import Counter
from hashlib import sha256
import json
import math
import random

from hephaestus.data.generated_validation import validate_original_example
from hephaestus.storage.verified import canonical_bytes

FAMILIES = ("arithmetic", "json", "symbolic", "reasoning")
CASES_PER_FAMILY = 512
CLAIM_SCOPE = "internal_prospective_original_task_only"
SCHEMA = "prospective-paired-json-v2"


def exact_mcnemar_p(wins, losses):
    """Two-sided exact matched-pair null probability; no continuity fudge."""
    if any(type(v) is not int or v < 0 for v in (wins, losses)):
        raise ValueError("discordant counts must be nonnegative integers")
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * (tail / (1 << n)))


def paired_binary_statistics(before, after, *, bootstrap_seed=152, bootstrap_draws=2000):
    if (len(before) != len(after) or not before or
            bootstrap_draws < 100 or type(bootstrap_seed) is not int or
            any(type(v) is not bool for v in list(before) + list(after))):
        raise ValueError("strict paired Boolean observations required")
    diffs = [int(a) - int(b) for b, a in zip(before, after, strict=True)]
    wins = diffs.count(1)
    losses = diffs.count(-1)
    n = len(diffs)
    generator = random.Random(bootstrap_seed)
    estimates = sorted(sum(diffs[generator.randrange(n)] for _ in range(n))/n
                       for _ in range(bootstrap_draws))
    lower = estimates[int(0.025 * bootstrap_draws)]
    upper = estimates[min(bootstrap_draws - 1, int(0.975 * bootstrap_draws))]
    return {
        "count": n,
        "baseline_correct": sum(before),
        "candidate_correct": sum(after),
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
        "paired_absolute_gain": sum(diffs) / n,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(wins, losses),
        "paired_bootstrap_95_ci": [lower, upper],
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
    }


def score_precommitted_holdout(rows, *, expected_commitment,
                               exposure_sample_ids, bootstrap_seed=152):
    """Score once against externally frozen item commitment and exposure ledger.

    Raw generations, target answers and external custodian signatures are NOT
    accepted as verified merely because this pure-Python function was called.
    The public summary intentionally omits all protected row content.
    """
    if (not isinstance(rows, list) or len(rows) != CASES_PER_FAMILY * len(FAMILIES)
            or not isinstance(expected_commitment, str)
            or not expected_commitment.startswith("sha256:")
            or len(expected_commitment) != 71
            or not isinstance(exposure_sample_ids, (set, frozenset))
            or not exposure_sample_ids):
        raise ValueError("complete externally precommitted holdout and exposure ledger required")
    before, after, fmt_before, fmt_after = [], [], [], []
    canonical, seen = [], set()
    counts, family_gain = Counter(), {key: [0, 0] for key in FAMILIES}
    eos_failures, truncations, raw_hashes = 0, 0, []
    for item in rows:
        if not isinstance(item, dict) or set(item) != {
            "prompt", "target", "family", "sample_id", "baseline_raw",
            "candidate_raw", "baseline_eos", "candidate_eos",
            "baseline_truncated", "candidate_truncated",
        }:
            raise ValueError("heldout row is incomplete or has unexpected fields")
        key = validate_original_example(item)
        sid = "sha256:" + sha256(canonical_bytes(key)).hexdigest()
        if (item["sample_id"] != sid or sid in seen or
                sid in exposure_sample_ids or item["family"] not in FAMILIES):
            raise ValueError("duplicate, exposed, wrong-family or altered heldout case")
        seen.add(sid)
        counts[item["family"]] += 1
        canonical.append({
            "sample_id": sid, "family": item["family"],
            "prompt": item["prompt"], "target": item["target"],
        })
        for attr in ("baseline_eos", "candidate_eos",
                     "baseline_truncated", "candidate_truncated"):
            if type(item[attr]) is not bool:
                raise ValueError("generation EOS/truncation receipt flags missing")
        baseline = score_prediction(item["baseline_raw"], item["target"])
        candidate = score_prediction(item["candidate_raw"], item["target"])
        before.append(baseline["exact_correct"])
        after.append(candidate["exact_correct"])
        fmt_before.append(baseline["valid_json"])
        fmt_after.append(candidate["valid_json"])
        family_gain[item["family"]][0] += int(baseline["exact_correct"])
        family_gain[item["family"]][1] += int(candidate["exact_correct"])
        eos_failures += int(not item["candidate_eos"])
        truncations += int(item["candidate_truncated"])
        raw_hashes.append({
            "sample_id": sid,
            "baseline_sha256": "sha256:" + sha256(item["baseline_raw"].encode()).hexdigest(),
            "candidate_sha256": "sha256:" + sha256(item["candidate_raw"].encode()).hexdigest(),
        })
    if any(counts[k] != CASES_PER_FAMILY for k in FAMILIES):
        raise ValueError("heldout families must have exactly 512 independent items each")
    commitment = "sha256:" + sha256(canonical_bytes(
        sorted(canonical, key=lambda item: item["sample_id"])
    )).hexdigest()
    if commitment != expected_commitment:
        raise PermissionError("heldout answers/items do not match external frozen commitment")
    effect = paired_binary_statistics(before, after, bootstrap_seed=bootstrap_seed)
    fmt_effect = paired_binary_statistics(fmt_before, fmt_after, bootstrap_seed=bootstrap_seed)
    losses = effect["losses"]
    gates = {
        "minimum_absolute_gain_5pp": effect["paired_absolute_gain"] >= 0.05,
        "exact_p_below_005": effect["exact_mcnemar_two_sided_p"] < 0.05,
        "bootstrap_lower_above_zero": effect["paired_bootstrap_95_ci"][0] > 0,
        "no_new_format_regression_in_aggregate": fmt_effect["paired_absolute_gain"] >= 0,
        "losses_at_most_one_percent": losses <= len(rows) // 100,
        "every_family_nonnegative": all(a <= b for a, b in family_gain.values()),
        "all_candidate_eos_terminated": eos_failures == 0,
        "no_candidate_truncated": truncations == 0,
    }
    return {
        "schema": SCHEMA,
        "scorer_version": SCORER_VERSION,
        "scope": CLAIM_SCOPE,
        "holdout_commitment": commitment,
        "known_exposure_count": len(exposure_sample_ids),
        "paired_exact": effect,
        "paired_valid_json": fmt_effect,
        "family_correctness": dict(family_gain),
        "deterministic_gates": gates,
        "raw_generation_sha256_by_case": sorted(raw_hashes, key=lambda r: r["sample_id"]),
        "internal_effect_supported": all(gates.values()),
        "external_certification": False,
        "model_promoted": False,
        "checkpoint_bytes_verified": False,
        "generation_receipts_independently_verified": False,
        "limitations": [
            "This pure scorer is not a model/holdout custodian or signed attestation.",
            "Matched outcomes apply to one original four-family task grammar only.",
            "Full external custody must verify original raw generations and identity.",
        ],
    }
