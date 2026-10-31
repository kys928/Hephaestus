"""Prospective distribution preflight and planning-only paired power simulation.

No confirmatory model is run and no key or signature is created here. The
holdout builder belongs on the independently operated custodian side.
"""
from hashlib import sha256
import math
import random

from hephaestus.data.generated_factory import _example
from hephaestus.data.generated_validation import validate_original_example
from hephaestus.schemas.prospective_study import ProspectiveDesign
from hephaestus.storage.verified import canonical_bytes
from .paired_holdout import exact_mcnemar_p


def arithmetic_capacity(difficulty: int) -> dict[str, int]:
    if type(difficulty) is not int or not 10 <= difficulty <= 1000:
        raise ValueError("bounded declared arithmetic difficulty required")
    counts = {"train": 0, "development": 0, "evaluation": 0}
    for operation in ("plus", "minus", "times"):
        for a in range(difficulty):
            for b in range(a if operation != "minus" else 0, difficulty):
                key = ("arithmetic", operation, (a, b))
                bucket = int(sha256(canonical_bytes(key)).hexdigest(), 16) % 10
                counts["development" if bucket == 8 else "evaluation" if bucket == 9 else "train"] += 1
    return counts


def build_custodian_holdout(design: ProspectiveDesign, *, custodian_seed: int,
                           exposure_sample_ids: set[str]) -> list[dict]:
    """Unique draws from the declared proposal distribution, conditional bucket9.

    Uses original v1 family proposals with range100, not a claim of uniform
    sampling over canonical equivalence classes. Exact/semantic aliases are
    rejected before retention. The seed must remain with the real custodian.
    """
    design.commitment()
    if type(custodian_seed) is not int or not exposure_sample_ids:
        raise ValueError("custodian seed and nonempty prior exposure ledger required")
    if arithmetic_capacity(design.difficulty)["evaluation"] < design.cases_per_family:
        raise ValueError("declared arithmetic distribution has insufficient canonical capacity")
    rng = random.Random(custodian_seed)
    seen, rows = set(exposure_sample_ids), []
    for family in design.families:
        accepted = 0
        for _ in range(design.cases_per_family * 2000):
            row = _example(rng, family, design.difficulty)
            key = validate_original_example(row)
            digest = sha256(canonical_bytes(key)).hexdigest()
            sid = "sha256:" + digest
            if int(digest, 16) % 10 != 9 or sid in seen:
                continue
            row["sample_id"] = sid
            rows.append(row)
            seen.add(sid)
            accepted += 1
            if accepted == design.cases_per_family:
                break
        else:
            raise ValueError("declared holdout distribution exhausted after exposure exclusions")
    return rows


def holdout_commitment(rows: list[dict]) -> str:
    items = []
    seen = set()
    for row in rows:
        sid = "sha256:" + sha256(canonical_bytes(validate_original_example(row))).hexdigest()
        if row.get("sample_id") != sid or sid in seen:
            raise ValueError("altered or duplicate canonical holdout identity")
        seen.add(sid)
        items.append({k: row[k] for k in ("sample_id", "family", "prompt", "target")})
    return "sha256:" + sha256(canonical_bytes(sorted(items, key=lambda r: r["sample_id"]))).hexdigest()


def planning_power(*, win_probability: float, loss_probability: float,
                   trials: int = 1000, seed: int = 152000,
                   assumed_family_icc: float = 0.0, eos_success_probability: float = 1.0) -> dict:
    """Monte Carlo design sensitivity, NOT an observed model effect.

    Counts are multinomial matched wins/losses/ties in four fixed strata.
    A Dirichlet mixture adds stated within-family ICC for sensitivity only.
    Each trial applies the actual gain, loss, family and exact-McNemar gates,
    plus a 2,000-draw stratified paired bootstrap. Format is conditioned valid;
    EOS factors are explicitly hypothetical independent-event calculations.
    """
    import numpy as np  # Optional planning dependency; no model downloads.
    if (not all(math.isfinite(v) for v in (win_probability, loss_probability,
            assumed_family_icc, eos_success_probability)) or min(win_probability, loss_probability) < 0
            or win_probability + loss_probability > 1 or not 0 <= assumed_family_icc < 1
            or not 0 <= eos_success_probability <= 1 or type(trials) is not int or trials < 100):
        raise ValueError("valid pre-test probabilities and at least 100 planning trials required")
    design = ProspectiveDesign()
    rng = np.random.default_rng(seed)
    probabilities = np.array([win_probability, loss_probability, 1-win_probability-loss_probability])
    success = 0
    for _ in range(trials):
        family_counts = []
        for _family in design.families:
            distribution = probabilities
            if assumed_family_icc:
                nonzero = probabilities > 0
                distribution = np.zeros(3)
                distribution[nonzero] = rng.dirichlet(probabilities[nonzero] * (1/assumed_family_icc - 1))
            family_counts.append(rng.multinomial(design.cases_per_family, distribution))
        counts = np.array(family_counts)
        wins, losses, _ties = counts.sum(axis=0)
        n = design.cases_per_family * 4
        preliminary = ((int(wins)-int(losses))/n >= design.minimum_gain
            and losses <= n // 100 and np.all(counts[:, 0] >= counts[:, 1])
            and exact_mcnemar_p(int(wins), int(losses)) < design.alpha)
        if not preliminary:
            continue
        estimates = np.zeros(design.bootstrap_draws, dtype=np.int64)
        for observed in counts:
            sample = rng.multinomial(design.cases_per_family, observed/design.cases_per_family,
                                     size=design.bootstrap_draws)
            estimates += sample[:, 0] - sample[:, 1]
        estimates.sort()
        success += int(estimates[int(.025 * design.bootstrap_draws)] > 0)
    rate = success / trials
    z = 1.959963984540054
    denominator = 1 + z*z/trials
    center = (rate + z*z/(2*trials)) / denominator
    radius = z * math.sqrt(rate*(1-rate)/trials + z*z/(4*trials*trials)) / denominator
    return {"schema": "prospective-design-power-sensitivity.v1", "planning_only": True,
        "design_commitment": design.commitment(), "cases": 2048,
        "win_probability": win_probability, "loss_probability": loss_probability,
        "true_gain_assumption": win_probability-loss_probability, "family_icc_assumption": assumed_family_icc,
        "trials": trials, "simulation_seed": seed, "statistical_gate_power": rate,
        "monte_carlo_wilson_95_interval": [max(0, center-radius), min(1, center+radius)],
        "hypothetical_eos_all_cases_probability_if_independent": eos_success_probability**n,
        "hypothetical_joint_power_if_independent_eos": rate * eos_success_probability**n,
        "limitations": ["Not measured model performance; probabilities are planning assumptions.",
            "Conditioned on nonregressing valid-JSON format and nontruncated generation.",
            "Four fixed families/templates cannot identify generalization variance to new templates.",
            "Positive ICC is sensitivity, not an empirically estimated correlation.",
            "Independent blinded pilot and custodian approval are still required before final freeze."]}
