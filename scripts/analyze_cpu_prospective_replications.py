#!/usr/bin/env python3
"""Aggregate authentic prospective CPU campaign reports without granting promotion."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics


def analyze_reports(paths):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    if len(reports) < 3:
        raise ValueError("at least three independently executed training seeds required")
    seeds, workflows, experiments, source_seeds, heldout, summaries = set(), set(), set(), set(), set(), []
    for report in reports:
        evidence = report.get("evidence") or {}
        science = evidence.get("prospective") or {}
        trained = evidence.get("actual_training") or {}
        if (report.get("status") != "completed" or report.get("completed_operations") != 45 or
                evidence.get("verified_operations") != 45 or evidence.get("verified_domain_operations") != 40 or
                evidence.get("claim_scope") != "internal_prospective_generated_task_research" or
                evidence.get("independent_scientific_certification") is not False or
                evidence.get("five_frozen_cuda_roles_executed") is not False or
                trained.get("completed_optimizer_steps") != 8 or
                trained.get("weights_changed") is not True or
                science.get("count") != 24 or science.get("target_nll_only_diagnostic") is not True):
            raise ValueError("missing eight-step genuine 45/45 evidence or honest research boundary")
        for name in ("correct_before", "correct_after", "json_before", "json_after"):
            if type(science.get(name)) is not int or not 0 <= science[name] <= science["count"]:
                raise ValueError("invalid task count: " + name)
        before, after = science.get("mean_target_nll_before"), science.get("mean_target_nll_after")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and
                   0 <= x < 1e6 for x in (before, after)):
            raise ValueError("held-out teacher-forced target NLL is missing or nonfinite")
        seed = evidence.get("cpu_seed")
        if type(seed) is not int or seed in seeds:
            raise ValueError("duplicate or invalid CPU training seed")
        seeds.add(seed)
        workflow, experiment = report.get("workflow_id"), evidence.get("experiment_id")
        if not workflow or workflow in workflows or not experiment or experiment in experiments:
            raise ValueError("experiment/workflow receipts not independent")
        workflows.add(workflow)
        experiments.add(experiment)
        source_seeds.add(evidence.get("dataset_seed"))
        heldout.add(science.get("evaluation_item_commitment"))
        summaries.append({"cpu_seed": seed, "workflow_id": workflow,
                          "optimizer_steps": trained["completed_optimizer_steps"],
                          "correct_before": science["correct_before"], "correct_after": science["correct_after"],
                          "json_before": science["json_before"], "json_after": science["json_after"],
                          "mean_target_nll_before": before, "mean_target_nll_after": after,
                          "target_nll_delta": after-before})
    if (len(source_seeds) != 1 or None in source_seeds or len(heldout) != 1 or
            not isinstance(next(iter(heldout)), str) or
            not next(iter(heldout)).startswith("sha256:")):
        raise ValueError("training seeds must share one frozen original-data seed and exact held-out item commitment")
    summaries.sort(key=lambda x: x["cpu_seed"])
    nll_deltas = [x["target_nll_delta"] for x in summaries]
    correctness = [x["correct_after"]-x["correct_before"] for x in summaries]
    return {"schema": "cpu-prospective-multiseed-summary.v1",
            "claim_scope": "internal_prospective_generated_task_research",
            "training_seed_count": len(summaries),
            "dataset_seed": next(iter(source_seeds)), "heldout_item_commitment": next(iter(heldout)),
            "per_seed": summaries, "mean_target_nll_delta": statistics.mean(nll_deltas),
            "sample_sd_target_nll_delta": statistics.stdev(nll_deltas),
            "nll_reduction_on_all_seeds": all(x < 0 for x in nll_deltas),
            "task_correctness_improvement_on_all_seeds": all(x > 0 for x in correctness),
            "model_promoted": False, "external_certification": False,
            "general_reasoning_improvement_claimed": False,
            "limitations": ["One original deterministic generated-task family mixture, not general-language pretraining.",
                            "Held-out likelihood reduction alone is not demonstrated answer correctness.",
                            "Three seeds are a small variance probe, not external scientific certification."]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("reports", type=Path, nargs="+")
    args = ap.parse_args()
    result = analyze_reports(args.reports)
    if args.output.exists():
        raise FileExistsError("refuse to overwrite original summary")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
