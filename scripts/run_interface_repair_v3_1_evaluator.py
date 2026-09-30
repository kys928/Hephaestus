#!/usr/bin/env python3
"""V3.1 Evaluator execution: fresh preflight split + one bounded JSON-format retry.

The retry is format-only: it receives the same evidence and machine state, no target
labels, and cannot alter the deterministic Evaluator action boundary. Scientific
and system certification gates remain unchanged.
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import run_interface_repair_v3_evaluator as v3

_ORIGINAL_GENERATE = v3.v2.generate
_ORIGINAL_SUMMARIZE = v3.summarize_v3


def generate_v31(
    model: Any,
    tokenizer: Any,
    runtime: Mapping[str, Any],
    text: str,
    seed: int,
    deadline: float,
) -> dict[str, Any]:
    first = _ORIGINAL_GENERATE(model, tokenizer, runtime, text, seed, deadline)
    first["format_retry_attempted"] = False
    first["format_retry_succeeded"] = False
    first["initial_complete_contract_json"] = bool(first["complete_contract_json"])
    first["initial_generated_tokens"] = int(first["generated_tokens"])
    first["initial_latency_seconds"] = float(first["latency_seconds"])
    if first["complete_contract_json"] or first["deadline_hit"]:
        return first

    retry_prompt = text + """

FORMAT-RETRY NOTICE:
Your immediately previous generation did not satisfy the required JSON contract.
Re-evaluate the SAME verified evidence and machine state. Do not invent evidence or
change scientific state merely to satisfy formatting. Output exactly one complete,
valid JSON object and nothing else. Start with { and end with }. Include each of the
seven required keys exactly once. No markdown, preamble, suffix, or hidden reasoning.
"""
    retry = _ORIGINAL_GENERATE(model, tokenizer, runtime, retry_prompt, seed + 700_001, deadline)
    retry["format_retry_attempted"] = True
    retry["format_retry_succeeded"] = bool(retry["complete_contract_json"])
    retry["initial_complete_contract_json"] = False
    retry["initial_generated_tokens"] = int(first["generated_tokens"])
    retry["initial_latency_seconds"] = float(first["latency_seconds"])
    retry["generated_tokens"] = int(first["generated_tokens"]) + int(retry["generated_tokens"])
    retry["latency_seconds"] = float(first["latency_seconds"]) + float(retry["latency_seconds"])
    return retry


def summarize_v31(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    summary = dict(_ORIGINAL_SUMMARIZE(rows))
    generations = [row["generation"] for row in rows]
    n = max(1, len(generations))
    attempted = sum(bool(g.get("format_retry_attempted")) for g in generations)
    succeeded = sum(bool(g.get("format_retry_succeeded")) for g in generations)
    initial_complete = sum(bool(g.get("initial_complete_contract_json", g.get("complete_contract_json"))) for g in generations)
    final_complete = sum(bool(g.get("complete_contract_json")) for g in generations)
    summary.update({
        "format_retry_attempt_rate": attempted / n,
        "format_retry_success_rate": succeeded / max(1, attempted),
        "initial_complete_contract_json_rate": initial_complete / n,
        "final_complete_contract_json_rate": final_complete / n,
    })
    return summary


def run_preflight_v31(
    *, cfg: Mapping[str, Any], pack: Mapping[str, Any], role: str, model: Any, tokenizer: Any,
    runtime: Mapping[str, Any], deadline: float, client: Any, prefix: str, heartbeat,
) -> dict[str, Any]:
    n_eval = int(cfg["preflight"]["evaluation_cases"])
    n_steps = int(cfg["preflight"]["optimizer_steps"])
    sample = list(pack["partitions"][role]["preflight"])
    if n_eval != len(sample):
        raise RuntimeError("V3.1 preflight case count must equal dedicated preflight partition")

    heartbeat("preflight_evaluation_started", sample_count=n_eval, partition="dedicated_preflight")
    rows, _ = v3.v2.evaluate(
        model, tokenizer, runtime, pack, sample, role, int(cfg["training"]["seed"]), deadline,
        client, prefix, "preflight_eval", heartbeat, int(cfg["execution"]["evaluation_shard_size"]),
    )
    eval_seconds = sum(float(r["generation"]["latency_seconds"]) for r in rows)
    complete_rate = sum(bool(r["generation"]["complete_contract_json"]) for r in rows) / max(1, len(rows))
    initial_complete_rate = sum(bool(r["generation"].get("initial_complete_contract_json")) for r in rows) / max(1, len(rows))
    retry_attempts = sum(bool(r["generation"].get("format_retry_attempted")) for r in rows)
    retry_successes = sum(bool(r["generation"].get("format_retry_succeeded")) for r in rows)
    deadline_hits = sum(bool(r["generation"]["deadline_hit"]) for r in rows)

    heartbeat("preflight_training_started", optimizer_steps=n_steps)
    training = v3.v2.train(
        model, tokenizer, runtime, pack, pack["partitions"][role]["train"], role, cfg, deadline, heartbeat,
        steps_override=n_steps,
    )
    mean_eval_seconds = eval_seconds / max(1, len(rows))
    projected_eval_seconds = mean_eval_seconds * (
        2 * int(cfg["pack"]["certification_cases_per_role"])
        + 2 * int(cfg["pack"]["regression_cases_per_role"])
    )
    projected_train_seconds = float(training["seconds_per_optimizer_step"]) * int(cfg["training"]["optimizer_steps"][role])
    projected_role_seconds = projected_eval_seconds + projected_train_seconds + 120.0
    approved = (
        projected_role_seconds <= float(cfg["preflight"]["maximum_projected_role_seconds"])
        and complete_rate >= float(cfg["preflight"]["require_complete_json_rate"])
        and (not cfg["preflight"].get("require_no_deadline_hits") or deadline_hits == 0)
    )
    return {
        "status": "completed",
        "role": role,
        "preflight_partition": "dedicated_preflight",
        "sampled_evaluation_cases": n_eval,
        "sampled_optimizer_steps": n_steps,
        "mean_evaluation_seconds_per_case": mean_eval_seconds,
        "mean_generated_tokens": sum(int(r["generation"]["generated_tokens"]) for r in rows) / max(1, len(rows)),
        "contract_early_stop_rate": sum(bool(r["generation"]["contract_early_stop"]) for r in rows) / max(1, len(rows)),
        "initial_complete_contract_json_rate": initial_complete_rate,
        "complete_contract_json_rate": complete_rate,
        "format_retry_attempt_rate": retry_attempts / max(1, len(rows)),
        "format_retry_success_rate": retry_successes / max(1, retry_attempts),
        "deadline_hits": deadline_hits,
        "training": training,
        "projected_full_role_evaluation_seconds": projected_eval_seconds,
        "projected_full_role_training_seconds": projected_train_seconds,
        "projected_full_role_seconds": projected_role_seconds,
        "maximum_projected_role_seconds": cfg["preflight"]["maximum_projected_role_seconds"],
        "approved_for_full_run": approved,
    }


v3.v2.generate = generate_v31
v3.v1.summarize = summarize_v31
v3.v2.run_preflight = run_preflight_v31


if __name__ == "__main__":
    raise SystemExit(v3.v2.main())
