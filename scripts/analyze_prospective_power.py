"""Public planning sensitivity only: no task examples or protected-data access."""
import json

from hephaestus.evaluation.pilot_power import (
    analytic_paired_gate_power, all_eos_gate_sensitivity, exact_binomial_interval,
)


def analysis():
    rows = []
    for n in (256, 512, 1024, 2048):
        for gain in (.05, .08, .10):
            for loss in (.005, .01):
                for alpha in (.05, .05 / 3):
                    result = analytic_paired_gate_power(
                        n=n, win_probability=gain + loss, loss_probability=loss, alpha=alpha)
                    rows.append(dict(result, true_gain=gain, alpha=alpha))
    return {
        "schema": "analytic-paired-planning.v1",
        "scope": "IID upper bound for a subset of the original v2 gates; not final design acceptance",
        "power": rows,
        "eos_sensitivity": {str(n): all_eos_gate_sensitivity(n) for n in (256, 512, 1024, 2048)},
        "zero_losses_in_64_interval": exact_binomial_interval(0, 64),
        "three_successes_in_three_seeds_interval_if_iid": exact_binomial_interval(3, 3),
        "protected_final_material_accessed": False,
    }


if __name__ == "__main__":
    print(json.dumps(analysis(), indent=2, allow_nan=False))
