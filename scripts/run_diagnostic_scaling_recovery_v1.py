#!/usr/bin/env python3
"""Run one Diagnostic Scaling Recovery V1 candidate on ephemeral GPU storage.

Recovery V1 keeps the original LoRA training geometry but separates evaluation into:
1) fixed_non_thinking: frozen 256/96-token deterministic budgets, only for models
   with a documented hard native non-thinking switch; truncation is an efficiency
   failure in this lane, not a general capability conclusion.
2) reasoning_aware: native reasoning policy with adaptive token ladders. A sample
   that hits the current budget is retried with the same seed. Exhausting the
   ladder is inconclusive and is never converted into a floor capability score.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_adaptation_elasticity_v1 as elastic
import run_cognitive_topology_v1 as topology_runner
import run_diagnostic_scaling_v1 as base
from hephaestus.scoring.behavioral import evaluate_behavioral_sample

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/experiments/hephaestus_diagnostic_scaling_recovery_v1.json"
TOPOLOGY_PATH = ROOT / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"
SEMANTIC_PATH = ROOT / "configs/eval_packs/semantic_behavior_v1.yaml"
SCIENTIFIC_PREFIX = "hephaestus/scientific/v1"
EPHEMERAL_ROOT = Path("/opt/hephaestus-diagnostic-scaling-recovery")
LANES = ("fixed_non_thinking", "reasoning_aware")


def progress_checkpoint_prefix(contract: dict[str, Any], contract_sha: str, model_id: str) -> str:
    settings = contract.get("execution", {}).get("progress_checkpointing", {})
    if settings.get("enabled") is not True:
        raise RuntimeError("recovery progress checkpointing must remain enabled")
    series_id = str(settings.get("series_id") or "").strip()
    if not series_id:
        raise RuntimeError("recovery progress checkpoint series_id is missing")
    return f"{SCIENTIFIC_PREFIX}/diagnostic_scaling_recovery_progress/{series_id}/{contract_sha}/{base.slug(model_id)}"


def upload_file_verified(client: Any, source: Path, key: str) -> dict[str, Any]:
    client.upload_file(str(source), base.bucket(), key)
    return base.verify_s3_file(client, key, source)


def sync_tree_verified(client: Any, root: Path, prefix: str) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return base.upload_tree_verified(client, root, prefix)


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def lane_eligible(candidate: dict[str, Any], lane: str) -> bool:
    return bool(candidate[lane].get("eligible"))


def project_reasoning(raw: str, contract: dict[str, Any], lane: str) -> str:
    if lane == "fixed_non_thinking":
        return raw.strip()
    policy = contract["evaluation_lanes"]["reasoning_aware"]
    best_end = -1
    for delimiter in policy["reasoning_close_delimiters"]:
        index = raw.rfind(str(delimiter))
        if index >= 0:
            best_end = max(best_end, index + len(str(delimiter)))
    return (raw[best_end:] if best_end >= 0 else raw).strip()


def render_prompt(tokenizer: Any, prompt: str, candidate: dict[str, Any], lane: str) -> str:
    cfg = candidate[lane]
    kwargs = dict(cfg.get("chat_template_kwargs") or {})
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    prefill = cfg.get("assistant_prefill")
    if prefill:
        rendered += str(prefill)
    return rendered


def generate_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    candidate: dict[str, Any],
    lane: str,
    seed: int,
    max_new_tokens: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class FirstTokenClock(StoppingCriteria):
        def __init__(self) -> None:
            self.first_time: float | None = None

        def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
            if self.first_time is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.first_time = time.perf_counter()
            return False

    rendered = render_prompt(tokenizer, prompt, candidate, lane)
    encoded = tokenizer(rendered, return_tensors="pt", truncation=False)
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[1])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    clock = FirstTokenClock()
    started = time.perf_counter()

    if lane == "fixed_non_thinking":
        lane_cfg = contract["evaluation_lanes"][lane]
        generation_kwargs: dict[str, Any] = {"do_sample": False}
    else:
        lane_cfg = candidate[lane]["decoding"]
        generation_kwargs = {
            "do_sample": bool(lane_cfg["do_sample"]),
            "temperature": float(lane_cfg["temperature"]),
            "top_p": float(lane_cfg["top_p"]),
        }
        if lane_cfg.get("top_k") is not None:
            generation_kwargs["top_k"] = int(lane_cfg["top_k"])

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            **generation_kwargs,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([clock]),
            return_dict_in_generate=True,
            use_cache=True,
        )
    torch.cuda.synchronize()
    finished = time.perf_counter()
    sequence = generated.sequences[0]
    continuation = sequence[input_width:]
    text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()
    finish_reason = (
        "eos" if len(continuation) and int(continuation[-1]) in eos_ids
        else "max_tokens" if len(continuation) >= int(max_new_tokens)
        else "stopped"
    )
    generated_tokens = int(continuation.shape[0])
    total = finished - started
    ttft = (clock.first_time - started) if clock.first_time is not None else total
    return {
        "output": text,
        "prompt_tokens": int(encoded["attention_mask"].sum().item()),
        "generated_tokens": generated_tokens,
        "max_new_tokens": int(max_new_tokens),
        "finish_reason": finish_reason,
        "ttft_seconds": ttft,
        "total_latency_seconds": total,
        "tokens_per_second": generated_tokens / total if total > 0 else 0.0,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "allocated_vram_bytes_after": int(torch.cuda.memory_allocated()),
    }


def topology_complete(projected: str) -> bool:
    _payload, valid = topology_runner._strict_parse(projected)
    return bool(valid)


def adaptive_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    candidate: dict[str, Any],
    lane: str,
    seed: int,
    token_ladder: list[int],
    contract: dict[str, Any],
    require_schema: bool,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for index, budget in enumerate(token_ladder):
        generated = generate_once(
            model,
            tokenizer,
            prompt,
            candidate=candidate,
            lane=lane,
            seed=seed,
            max_new_tokens=int(budget),
            contract=contract,
        )
        raw = str(generated["output"])
        projected = project_reasoning(raw, contract, lane)
        complete_schema = topology_complete(projected) if require_schema else True
        needs_retry = (
            generated["finish_reason"] == "max_tokens"
            or (require_schema and not complete_schema)
        )
        attempts.append({
            "budget": int(budget),
            "finish_reason": generated["finish_reason"],
            "generated_tokens": generated["generated_tokens"],
            "projected_schema_complete": complete_schema if require_schema else None,
        })
        if needs_retry and index < len(token_ladder) - 1:
            continue
        exhausted = bool(needs_retry and index == len(token_ladder) - 1)
        return {
            **generated,
            "output": raw,
            "projected_output": projected,
            "budget_attempts": attempts,
            "budget_retry_count": len(attempts) - 1,
            "budget_exhausted": exhausted,
            "final_schema_complete": complete_schema if require_schema else None,
        }
    raise RuntimeError("adaptive generation ladder unexpectedly produced no result")


def evaluate_role_lane(
    model: Any,
    tokenizer: Any,
    topology: dict[str, Any],
    contract: dict[str, Any],
    candidate: dict[str, Any],
    role: str,
    seeds: list[int],
    lane: str,
    sample_path: Path,
) -> dict[str, Any]:
    if not lane_eligible(candidate, lane):
        return {"status": "not_applicable", "reason": candidate[lane].get("reason"), "summary": None, "samples": []}

    role_cases = [case for case in topology["cases"] if case["role"] == role]
    role_spec = dict(topology)
    role_spec["roles"] = [role]
    role_spec["generation"] = dict(topology["generation"])
    role_spec["generation"]["seeds"] = list(seeds)
    role_spec["generation"]["repetitions"] = len(seeds)
    samples: list[dict[str, Any]] = []
    scored_samples: list[dict[str, Any]] = []
    inconclusive = 0
    fixed_truncations = 0
    tokenizer.padding_side = "left"
    model.eval()

    for seed in seeds:
        for case in role_cases:
            prompt = topology_runner._prompt(topology, case)
            if lane == "fixed_non_thinking":
                generated = generate_once(
                    model,
                    tokenizer,
                    prompt,
                    candidate=candidate,
                    lane=lane,
                    seed=int(seed),
                    max_new_tokens=int(contract["evaluation_lanes"][lane]["topology_max_new_tokens"]),
                    contract=contract,
                )
                raw_output = str(generated["output"])
                scored_output = project_reasoning(raw_output, contract, lane)
                fixed_truncations += int(generated["finish_reason"] == "max_tokens")
                score = topology_runner.score_response(topology, case, scored_output)
                status = "scored"
            else:
                generated = adaptive_generate(
                    model,
                    tokenizer,
                    prompt,
                    candidate=candidate,
                    lane=lane,
                    seed=int(seed),
                    token_ladder=[int(value) for value in candidate[lane]["topology_token_ladder"]],
                    contract=contract,
                    require_schema=True,
                )
                raw_output = str(generated["output"])
                scored_output = str(generated["projected_output"])
                if generated["budget_exhausted"]:
                    score = None
                    status = "inconclusive_reasoning_budget_exhausted"
                    inconclusive += 1
                else:
                    score = topology_runner.score_response(topology, case, scored_output)
                    status = "scored"

            row = {
                "model_id": candidate["model_id"],
                "revision": candidate["revision"],
                "lane": lane,
                "case_id": case["case_id"],
                "role": role,
                "condition": case["condition"],
                "pair_id": case.get("pair_id"),
                "seed": int(seed),
                "status": status,
                "output": scored_output,
                "raw_output": raw_output,
                "reasoning_projection_applied": scored_output != raw_output.strip(),
                "generation": {key: value for key, value in generated.items() if key not in {"output", "projected_output"}},
                "score": score,
            }
            samples.append(row)
            base.append_jsonl(sample_path, row)
            if score is not None:
                scored_samples.append(row)

    expected = len(role_cases) * len(seeds)
    if len(samples) != expected:
        raise RuntimeError(f"incomplete role evidence: {len(samples)} != {expected}")
    if scored_samples:
        summary = topology_runner._summarize_model(role_spec, candidate, scored_samples, {})["roles"][role]
    else:
        summary = {
            "quality": None,
            "quality_100": None,
            "schema_compliance": 0.0,
            "hallucination_rate": 0.0,
            "sample_count": 0,
            "case_count": len(role_cases),
        }
    summary = dict(summary)
    summary.update({
        "lane": lane,
        "expected_sample_count": expected,
        "scored_sample_count": len(scored_samples),
        "inconclusive_sample_count": inconclusive,
        "fixed_budget_truncation_count": fixed_truncations,
        "claimable": inconclusive == 0,
    })
    tokenizer.padding_side = "right"
    return {"status": "complete" if inconclusive == 0 else "inconclusive", "summary": summary, "samples": samples}


def evaluate_semantic_lane(
    model: Any,
    tokenizer: Any,
    semantic: dict[str, Any],
    contract: dict[str, Any],
    candidate: dict[str, Any],
    seeds: list[int],
    lane: str,
    sample_path: Path,
) -> dict[str, Any]:
    if not lane_eligible(candidate, lane):
        return {"status": "not_applicable", "reason": candidate[lane].get("reason"), "summary": None, "samples": []}

    tasks = base.semantic_tasks(semantic)
    rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    hard_failures: set[str] = set()
    inconclusive = 0
    fixed_truncations = 0
    tokenizer.padding_side = "left"
    model.eval()

    for seed in seeds:
        for task in tasks:
            prompt = str(task["prompt"])
            if lane == "fixed_non_thinking":
                generated = generate_once(
                    model,
                    tokenizer,
                    prompt,
                    candidate=candidate,
                    lane=lane,
                    seed=int(seed),
                    max_new_tokens=int(contract["evaluation_lanes"][lane]["semantic_max_new_tokens"]),
                    contract=contract,
                )
                raw_output = str(generated["output"])
                scored_output = project_reasoning(raw_output, contract, lane)
                fixed_truncations += int(generated["finish_reason"] == "max_tokens")
                score = evaluate_behavioral_sample(task, scored_output, seed)
                status = "scored"
            else:
                generated = adaptive_generate(
                    model,
                    tokenizer,
                    prompt,
                    candidate=candidate,
                    lane=lane,
                    seed=int(seed),
                    token_ladder=[int(value) for value in candidate[lane]["semantic_token_ladder"]],
                    contract=contract,
                    require_schema=False,
                )
                raw_output = str(generated["output"])
                scored_output = str(generated["projected_output"])
                if generated["budget_exhausted"]:
                    score = None
                    status = "inconclusive_reasoning_budget_exhausted"
                    inconclusive += 1
                else:
                    score = evaluate_behavioral_sample(task, scored_output, seed)
                    status = "scored"

            score_dict = None if score is None else score.to_dict()
            if score is not None:
                for check in score.checks:
                    if check.hard and not check.passed:
                        hard_failures.add(f"{task['task_id']}:{seed}:{check.name}")
            row = {
                "model_id": candidate["model_id"],
                "revision": candidate["revision"],
                "lane": lane,
                "task_id": task["task_id"],
                "seed": int(seed),
                "status": status,
                "output": scored_output,
                "raw_output": raw_output,
                "reasoning_projection_applied": scored_output != raw_output.strip(),
                "generation": {key: value for key, value in generated.items() if key not in {"output", "projected_output"}},
                "score": score_dict,
            }
            rows.append(row)
            base.append_jsonl(sample_path, row)
            if score_dict is not None:
                scored_rows.append(row)

    expected = len(tasks) * len(seeds)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete semantic evidence: {len(rows)} != {expected}")
    summary = {
        "lane": lane,
        "sample_count": len(rows),
        "scored_sample_count": len(scored_rows),
        "inconclusive_sample_count": inconclusive,
        "fixed_budget_truncation_count": fixed_truncations,
        "task_count": len(tasks),
        "mean_score": mean([float(row["score"]["overall_score"]) for row in scored_rows]),
        "deterministic_pass_rate": mean([float(row["score"]["deterministic_passed"]) for row in scored_rows]),
        "hard_failures": sorted(hard_failures),
        "hard_failure_count": len(hard_failures),
        "reasoning_projection_count": sum(int(row["reasoning_projection_applied"]) for row in rows),
        "claimable": inconclusive == 0,
    }
    tokenizer.padding_side = "right"
    return {"status": "complete" if inconclusive == 0 else "inconclusive", "summary": summary, "samples": rows}


def safety_for_lane(
    role_eval: dict[str, Any],
    semantic_eval: dict[str, Any],
    baseline_semantic: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if role_eval["status"] == "not_applicable" or semantic_eval["status"] == "not_applicable":
        return {"claimable": False, "status": "not_applicable", "interface_ok": False, "strict_safe": False, "bounded_safe": False}
    if not role_eval["summary"].get("claimable", True) or not semantic_eval["summary"].get("claimable", True) or not baseline_semantic.get("claimable", True):
        return {
            "claimable": False,
            "status": "inconclusive_reasoning_budget_exhausted",
            "interface_ok": False,
            "strict_safe": False,
            "bounded_safe": False,
            "new_hard_failures": [],
            "semantic_score_delta": None,
        }
    state = base.safe_state(role_eval["summary"], semantic_eval["summary"], baseline_semantic, contract)
    state["claimable"] = True
    state["status"] = "complete"
    return state


def lane_points(points: list[dict[str, Any]], lane: str) -> list[dict[str, Any]]:
    out = []
    for point in points:
        value = point["lanes"].get(lane)
        if not value or value.get("status") == "not_applicable" or value.get("role") is None:
            continue
        if not value.get("safety", {}).get("claimable", False):
            continue
        out.append({
            "optimizer_steps": point["optimizer_steps"],
            "approx_epochs": point["approx_epochs"],
            "role": value["role"],
            "semantic": value["semantic"],
            "safety": value["safety"],
            "training": point["training"],
        })
    return out


def lane_selections(points: list[dict[str, Any]], contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    selections: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    for lane in LANES:
        rows = lane_points(points, lane)
        if not rows:
            selections[lane] = None
            thresholds[lane] = None
            continue
        selections[lane] = base.select_points(rows)
        thresholds[lane] = base.threshold_distances(rows, [int(v) for v in contract["evaluation"]["quality_thresholds_100"]])
    return selections, thresholds


def evaluate_all_lanes(
    model: Any,
    tokenizer: Any,
    *,
    topology: dict[str, Any],
    semantic: dict[str, Any],
    contract: dict[str, Any],
    candidate: dict[str, Any],
    role: str,
    role_seeds: list[int],
    semantic_seeds: list[int],
    baseline_semantics: dict[str, Any],
    sample_root: Path,
    suffix: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for lane in LANES:
        if not lane_eligible(candidate, lane):
            result[lane] = {"status": "not_applicable", "reason": candidate[lane].get("reason"), "role": None, "semantic": None, "safety": {"claimable": False, "status": "not_applicable"}}
            continue
        role_eval = evaluate_role_lane(
            model, tokenizer, topology, contract, candidate, role, role_seeds, lane,
            sample_root / lane / f"role_{suffix}.jsonl",
        )
        sem_eval = evaluate_semantic_lane(
            model, tokenizer, semantic, contract, candidate, semantic_seeds, lane,
            sample_root / lane / f"semantic_{suffix}.jsonl",
        )
        safety = safety_for_lane(role_eval, sem_eval, baseline_semantics[lane], contract)
        result[lane] = {
            "status": "complete" if safety.get("claimable") else safety.get("status", "inconclusive"),
            "role": role_eval["summary"],
            "semantic": sem_eval["summary"],
            "safety": safety,
        }
    return result


def adapter_manifest(
    adapter_dir: Path,
    *,
    candidate: dict[str, Any],
    role: str,
    optimizer_steps: int,
    contract_sha: str,
    dataset_sha: str,
    trainable_parameters: int,
    target_modules: list[str],
    target_parameters: list[str],
) -> dict[str, Any]:
    manifest = base.adapter_manifest(
        adapter_dir,
        candidate=candidate,
        role=role,
        optimizer_steps=optimizer_steps,
        contract_sha=contract_sha,
        dataset_sha=dataset_sha,
        trainable_parameters=trainable_parameters,
        target_modules=target_modules,
        target_parameters=target_parameters,
    )
    manifest["manifest_version"] = "diagnostic-scaling-recovery-adapter.v1"
    return manifest


def train_role_curve(
    *,
    snapshot: Path,
    candidate: dict[str, Any],
    role: str,
    role_index: int,
    rows: list[dict[str, Any]],
    contract: dict[str, Any],
    contract_sha: str,
    topology: dict[str, Any],
    semantic: dict[str, Any],
    baseline_semantics: dict[str, Any],
    evidence_root: Path,
    dataset_sha: str,
    client: Any,
    checkpoint_prefix: str,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import get_linear_schedule_with_warmup

    model = tokenizer = None
    role_root = evidence_root / "roles" / role
    role_root.mkdir(parents=True, exist_ok=False)
    try:
        model, tokenizer, runtime = base.load_base(snapshot, candidate, contract)
        topology_runner._warmup(model, tokenizer)
        target_modules, target_parameters, rank_pattern = base.target_surface(model, candidate, contract)
        base.atomic_json(role_root / "runtime.json", runtime)
        base.atomic_json(role_root / "target_surface.json", {
            "target_modules": target_modules,
            "target_parameters": target_parameters,
            "rank_pattern": rank_pattern,
        })
        checkpoint_role_prefix = f"{checkpoint_prefix}/roles/{role}"
        upload_file_verified(client, role_root / "runtime.json", f"{checkpoint_role_prefix}/runtime.json")
        upload_file_verified(client, role_root / "target_surface.json", f"{checkpoint_role_prefix}/target_surface.json")

        role_seeds = [int(contract["evaluation"]["screening_seed"])]
        semantic_seeds = [int(v) for v in semantic["decoding_config"]["seeds"]]
        baseline_lanes = evaluate_all_lanes(
            model,
            tokenizer,
            topology=topology,
            semantic=semantic,
            contract=contract,
            candidate=candidate,
            role=role,
            role_seeds=role_seeds,
            semantic_seeds=semantic_seeds,
            baseline_semantics=baseline_semantics,
            sample_root=role_root / "samples",
            suffix="step_000",
        )
        points: list[dict[str, Any]] = [{
            "optimizer_steps": 0,
            "approx_epochs": 0.0,
            "lanes": baseline_lanes,
            "training": {
                "training_seconds_cumulative": 0.0,
                "supervised_tokens_cumulative": 0,
                "trainable_parameters": 0,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            },
            "adapter": None,
        }]

        train = contract["training"]
        torch.manual_seed(int(train["model_seed"]) + role_index)
        torch.cuda.manual_seed_all(int(train["model_seed"]) + role_index)
        shuffled = list(rows)
        random.Random(int(train["shuffle_seed"]) + role_index).shuffle(shuffled)
        encoded = [elastic._tokenize_example(tokenizer, row, int(train["max_seq_length"])) for row in shuffled]
        lora_kwargs: dict[str, Any] = {
            "r": int(train["rank"]),
            "lora_alpha": int(train["alpha"]),
            "lora_dropout": float(train["dropout"]),
            "bias": str(train["bias"]),
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": target_modules,
        }
        if target_parameters:
            lora_kwargs["target_parameters"] = target_parameters
            lora_kwargs["rank_pattern"] = rank_pattern
        peft_model = get_peft_model(model, LoraConfig(**lora_kwargs))
        model = None
        trainable_parameters = sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
        if trainable_parameters <= 0:
            raise RuntimeError("LoRA target surface produced no trainable parameters")
        router_names = [name for name, parameter in peft_model.named_parameters() if parameter.requires_grad and (name.endswith(".gate.weight") or ".gate.weight" in name)]
        if router_names:
            raise RuntimeError(f"MoE router unexpectedly became trainable: {router_names[:5]}")

        optimizer = torch.optim.AdamW(
            [parameter for parameter in peft_model.parameters() if parameter.requires_grad],
            lr=float(train["learning_rate"]),
            weight_decay=float(train["weight_decay"]),
        )
        max_steps = max(int(value) for value in train["dose_optimizer_steps"])
        warmup_steps = max(1, int(round(max_steps * float(train["warmup_ratio"]))))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps)
        accumulation = int(train["gradient_accumulation_steps"])
        dose_steps = {int(value) for value in train["dose_optimizer_steps"]}
        optimizer.zero_grad(set_to_none=True)
        optimizer_step = 0
        cumulative_tokens = 0
        cumulative_seconds = 0.0
        segment_losses: list[float] = []
        torch.cuda.reset_peak_memory_stats()

        stop = False
        outputs = None
        for _epoch in range(int(train["max_epochs"])):
            if stop:
                break
            for index, item in enumerate(encoded):
                started = time.perf_counter()
                peft_model.train()
                tokenizer.padding_side = "right"
                batch = elastic._tensorize(item)
                cumulative_tokens += sum(1 for label in item["labels"] if label != -100)
                outputs = peft_model(**batch)
                loss = outputs.loss / accumulation
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"non-finite LoRA loss for {candidate['model_id']} role={role}")
                loss.backward()
                segment_losses.append(float(loss.detach().item()) * accumulation)
                if (index + 1) % accumulation != 0:
                    cumulative_seconds += time.perf_counter() - started
                    continue
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in peft_model.parameters() if parameter.requires_grad],
                    float(train["max_grad_norm"]),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                cumulative_seconds += time.perf_counter() - started
                optimizer_step += 1
                if optimizer_step not in dose_steps:
                    continue

                adapter_dir = role_root / "adapters" / f"step_{optimizer_step:03d}"
                peft_model.save_pretrained(adapter_dir, safe_serialization=True)
                manifest = adapter_manifest(
                    adapter_dir,
                    candidate=candidate,
                    role=role,
                    optimizer_steps=optimizer_step,
                    contract_sha=contract_sha,
                    dataset_sha=dataset_sha,
                    trainable_parameters=trainable_parameters,
                    target_modules=target_modules,
                    target_parameters=target_parameters,
                )
                base.write_once_json(adapter_dir / "hephaestus_diagnostic_scaling_recovery_adapter_manifest.json", manifest)
                lane_results = evaluate_all_lanes(
                    peft_model,
                    tokenizer,
                    topology=topology,
                    semantic=semantic,
                    contract=contract,
                    candidate=candidate,
                    role=role,
                    role_seeds=role_seeds,
                    semantic_seeds=semantic_seeds,
                    baseline_semantics=baseline_semantics,
                    sample_root=role_root / "samples",
                    suffix=f"step_{optimizer_step:03d}",
                )
                point = {
                    "optimizer_steps": optimizer_step,
                    "approx_epochs": optimizer_step / float(train["expected_optimizer_steps_per_epoch"]),
                    "lanes": lane_results,
                    "training": {
                        "training_seconds_cumulative": cumulative_seconds,
                        "supervised_tokens_cumulative": cumulative_tokens,
                        "trainable_parameters": trainable_parameters,
                        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                        "mean_loss_since_previous_dose": mean(segment_losses),
                    },
                    "adapter": manifest,
                }
                points.append(point)
                segment_losses = []
                progress_dir = role_root / "progress"
                progress_dir.mkdir(parents=True, exist_ok=True)
                progress_path = progress_dir / f"step_{optimizer_step:03d}.json"
                base.write_once_json(progress_path, {
                    "result_version": "diagnostic-scaling-recovery-progress.v1",
                    "protocol_sha256": contract_sha,
                    "training_dataset_sha256": dataset_sha,
                    "model_id": candidate["model_id"],
                    "revision": candidate["revision"],
                    "role": role,
                    "optimizer_steps": optimizer_step,
                    "point": point,
                    "promotion_performed": False,
                    "lineage_mutated": False,
                })
                upload_file_verified(client, progress_path, f"{checkpoint_role_prefix}/progress/{progress_path.name}")
                sync_tree_verified(client, adapter_dir, f"{checkpoint_role_prefix}/adapters/{adapter_dir.name}")
                sync_tree_verified(client, role_root / "samples", f"{checkpoint_role_prefix}/samples")
                print("DIAGNOSTIC_SCALING_RECOVERY_CHECKPOINT_JSON " + json.dumps({
                    "model_id": candidate["model_id"],
                    "role": role,
                    "optimizer_steps": optimizer_step,
                    "checkpoint_prefix": checkpoint_role_prefix,
                }, sort_keys=True), flush=True)
                print("DIAGNOSTIC_SCALING_RECOVERY_DOSE_JSON " + json.dumps({
                    "model_id": candidate["model_id"],
                    "role": role,
                    "optimizer_steps": optimizer_step,
                    "lanes": {
                        lane: {
                            "status": lane_results[lane]["status"],
                            "quality_100": None if lane_results[lane].get("role") is None else lane_results[lane]["role"].get("quality_100"),
                            "strict_safe": lane_results[lane].get("safety", {}).get("strict_safe"),
                        }
                        for lane in LANES
                    },
                }, sort_keys=True), flush=True)
                if optimizer_step >= max_steps:
                    stop = True
                    break
        if optimizer_step != max_steps:
            raise RuntimeError(f"LoRA trajectory stopped at {optimizer_step} != {max_steps}")

        selections, thresholds = lane_selections(points, contract)
        selected_steps: set[int] = set()
        for lane in LANES:
            selection = selections.get(lane) or {}
            for step in selection.values():
                if step is not None and int(step) > 0:
                    selected_steps.add(int(step))
            lane_thresholds = thresholds.get(lane) or {}
            for mode in lane_thresholds.values():
                for row in mode.values():
                    if row and int(row["optimizer_steps"]) > 0:
                        selected_steps.add(int(row["optimizer_steps"]))

        del optimizer, scheduler
        if outputs is not None:
            del outputs
        try:
            base_model = peft_model.unload()
            del peft_model, base_model
        except Exception:
            del peft_model
        base.unload(tokenizer=tokenizer)
        tokenizer = None

        confirmations: dict[str, Any] = {lane: {} for lane in LANES}
        confirmation_seeds = [int(value) for value in contract["evaluation"]["full_confirmation_seeds"]]
        for lane in LANES:
            selection = selections.get(lane) or {}
            confirmation_steps = sorted({int(step) for step in selection.values() if step is not None})
            for step in confirmation_steps:
                confirm_model = confirm_tokenizer = None
                try:
                    confirm_model, confirm_tokenizer, _runtime = base.load_base(snapshot, candidate, contract)
                    if step > 0:
                        confirm_model = PeftModel.from_pretrained(
                            confirm_model,
                            role_root / "adapters" / f"step_{step:03d}",
                            is_trainable=False,
                        )
                    full = evaluate_role_lane(
                        confirm_model,
                        confirm_tokenizer,
                        topology,
                        contract,
                        candidate,
                        role,
                        confirmation_seeds,
                        lane,
                        role_root / "confirmations" / lane / f"role_step_{step:03d}.jsonl",
                    )
                    confirmations[lane][str(step)] = full["summary"]
                finally:
                    base.unload(confirm_model, confirm_tokenizer)

        adapters_root = role_root / "adapters"
        retained = []
        for adapter_dir in sorted(adapters_root.glob("step_*")):
            step = int(adapter_dir.name.split("_")[-1])
            if step in selected_steps:
                retained.append(step)
            else:
                shutil.rmtree(adapter_dir)

        curve = {
            "result_version": "diagnostic-scaling-recovery-role-curve.v1",
            "status": "complete",
            "protocol_sha256": contract_sha,
            "training_dataset_sha256": dataset_sha,
            "model_id": candidate["model_id"],
            "revision": candidate["revision"],
            "role": role,
            "points": points,
            "threshold_distances_by_lane": thresholds,
            "selections_by_lane": selections,
            "full_seed_confirmations_by_lane": confirmations,
            "retained_adapter_steps": retained,
            "target_module_count": len(target_modules),
            "target_parameter_count": len(target_parameters),
            "trainable_parameters": trainable_parameters,
            "cross_lane_ranking_performed": False,
            "promotion_performed": False,
            "lineage_mutated": False,
        }
        base.write_once_json(role_root / "role_curve.json", curve)
        sync_tree_verified(client, role_root, checkpoint_role_prefix)
        print("DIAGNOSTIC_SCALING_RECOVERY_ROLE_CHECKPOINT_JSON " + json.dumps({
            "model_id": candidate["model_id"],
            "role": role,
            "checkpoint_prefix": checkpoint_role_prefix,
            "status": "complete",
        }, sort_keys=True), flush=True)
        return curve
    finally:
        base.unload(model, tokenizer)


def main() -> int:
    import torch

    run_id = required("HEPHAESTUS_DS_RUN_ID")
    execution_id = required("HEPHAESTUS_EXECUTION_ID")
    attempt = int(required("HEPHAESTUS_ATTEMPT"))
    repo_sha = required("HEPHAESTUS_REPO_SHA")
    model_id = required("HEPHAESTUS_MODEL_ID")
    contract_raw = CONTRACT_PATH.read_bytes()
    contract_sha = sha(contract_raw)
    contract = json.loads(contract_raw)
    topology = json.loads(TOPOLOGY_PATH.read_text(encoding="utf-8"))
    semantic = json.loads(SEMANTIC_PATH.read_text(encoding="utf-8"))
    candidate = next((row for row in contract["candidates"] if row["model_id"] == model_id), None)
    if candidate is None:
        raise RuntimeError(f"model is not in Diagnostic Scaling Recovery cohort: {model_id}")
    if contract["governance"].get("paid_launch_allowed_now") is not True:
        raise RuntimeError("paid Diagnostic Scaling Recovery launch is not authorized by the current contract")

    work_root = EPHEMERAL_ROOT / run_id / base.slug(model_id) / f"attempt-{attempt}"
    admission_root = work_root / "admission"
    model_root = work_root / "model"
    evidence_root = work_root / "evidence"
    if work_root.exists():
        shutil.rmtree(work_root)
    admission_root.mkdir(parents=True)
    evidence_root.mkdir(parents=True)

    client = base.s3_client()
    terminal_key = f"{SCIENTIFIC_PREFIX}/executions/{execution_id}/attempt-{attempt}/driver_result.json"
    result: dict[str, Any] = {
        "result_version": "diagnostic-scaling-recovery-driver.v1",
        "status": "running",
        "run_id": run_id,
        "execution_id": execution_id,
        "attempt": attempt,
        "repo_sha": repo_sha,
        "model_id": model_id,
        "revision": candidate["revision"],
        "protocol_id": contract["protocol_id"],
        "protocol_sha256": contract_sha,
        "network_volume_attached": False,
        "training_performed": False,
        "promotion_performed": False,
        "lineage_mutated": False,
    }

    try:
        admission_prefix = f"{SCIENTIFIC_PREFIX}/diagnostic_scaling_recovery/model_admission/{repo_sha}"
        for name in ("admission.json", "protocol.json", "training.jsonl", "contamination_report.json", "requirements.txt", "remote_models.json"):
            base.download_s3(client, f"{admission_prefix}/{name}", admission_root / name)
        admission = json.loads((admission_root / "admission.json").read_text(encoding="utf-8"))
        admitted_contract = (admission_root / "protocol.json").read_bytes()
        if admission.get("status") != "recovery_launch_inputs_admitted" or admission.get("repo_sha") != repo_sha:
            raise RuntimeError("Diagnostic Scaling Recovery launch admission identity mismatch")
        if admission.get("protocol_sha256") != contract_sha or admitted_contract != contract_raw:
            raise RuntimeError("runtime recovery contract differs from admitted contract bytes")
        if admission.get("paid_launch_authorized") is not True:
            raise RuntimeError("recovery admission does not authorize paid launch")
        if admission.get("candidate_revisions", {}).get(model_id) != candidate["revision"]:
            raise RuntimeError("candidate revision differs from recovery launch admission")
        dataset_raw = (admission_root / "training.jsonl").read_bytes()
        dataset_sha = sha(dataset_raw)
        if dataset_sha != admission.get("training_dataset_sha256") or dataset_sha != contract["sources"]["source_training_dataset_sha256"]:
            raise RuntimeError("runtime training dataset differs from admitted source")
        runtime_versions = base.verify_runtime_versions(admission_root, contract)
        rows = [json.loads(line) for line in dataset_raw.decode("utf-8").splitlines() if line.strip()]
        roles = [*contract["roles"]["primary"], *contract["roles"]["calibration"]]
        per_role = {role: [row for row in rows if row.get("role") == role] for role in roles}
        expected = int(contract["training"]["examples_per_role"])
        if any(len(role_rows) != expected for role_rows in per_role.values()):
            raise RuntimeError("runtime role training data is incomplete")

        snapshot, model_manifest = base.materialize_model(candidate, model_root)
        checkpoint_prefix = progress_checkpoint_prefix(contract, contract_sha, model_id)
        base.write_once_json(evidence_root / "model_manifest.json", model_manifest)
        base.write_once_json(evidence_root / "run_manifest.json", {
            "run_id": run_id,
            "execution_id": execution_id,
            "attempt": attempt,
            "repo_sha": repo_sha,
            "protocol_sha256": contract_sha,
            "training_dataset_sha256": dataset_sha,
            "model_id": model_id,
            "revision": candidate["revision"],
            "roles": roles,
            "runtime_versions": runtime_versions,
            "evaluation_lanes": list(LANES),
            "fixed_non_thinking_eligible": lane_eligible(candidate, "fixed_non_thinking"),
            "network_volume_attached": False,
        })
        upload_file_verified(client, evidence_root / "model_manifest.json", f"{checkpoint_prefix}/model_manifest.json")
        upload_file_verified(client, evidence_root / "run_manifest.json", f"{checkpoint_prefix}/run_manifest.json")

        baseline_model = baseline_tokenizer = None
        baseline_semantics: dict[str, Any] = {}
        try:
            baseline_model, baseline_tokenizer, baseline_runtime = base.load_base(snapshot, candidate, contract)
            topology_runner._warmup(baseline_model, baseline_tokenizer)
            semantic_seeds = [int(value) for value in semantic["decoding_config"]["seeds"]]
            for lane in LANES:
                if not lane_eligible(candidate, lane):
                    baseline_semantics[lane] = {"claimable": False, "status": "not_applicable", "hard_failures": [], "mean_score": 0.0}
                    continue
                sem_eval = evaluate_semantic_lane(
                    baseline_model,
                    baseline_tokenizer,
                    semantic,
                    contract,
                    candidate,
                    semantic_seeds,
                    lane,
                    evidence_root / "baseline" / lane / "semantic_samples.jsonl",
                )
                baseline_semantics[lane] = sem_eval["summary"]
                base.write_once_json(evidence_root / "baseline" / lane / "semantic_summary.json", sem_eval["summary"])
            base.write_once_json(evidence_root / "baseline" / "runtime.json", baseline_runtime)
            sync_tree_verified(client, evidence_root / "baseline", f"{checkpoint_prefix}/baseline")
            print("DIAGNOSTIC_SCALING_RECOVERY_BASELINE_CHECKPOINT_JSON " + json.dumps({
                "model_id": model_id,
                "checkpoint_prefix": f"{checkpoint_prefix}/baseline",
            }, sort_keys=True), flush=True)
        finally:
            base.unload(baseline_model, baseline_tokenizer)

        role_results: dict[str, Any] = {}
        for role_index, role in enumerate(roles):
            role_results[role] = train_role_curve(
                snapshot=snapshot,
                candidate=candidate,
                role=role,
                role_index=role_index,
                rows=per_role[role],
                contract=contract,
                contract_sha=contract_sha,
                topology=topology,
                semantic=semantic,
                baseline_semantics=baseline_semantics,
                evidence_root=evidence_root,
                dataset_sha=dataset_sha,
                client=client,
                checkpoint_prefix=checkpoint_prefix,
            )
            print("DIAGNOSTIC_SCALING_RECOVERY_ROLE_COMPLETE_JSON " + json.dumps({
                "model_id": model_id,
                "role": role,
                "selections_by_lane": role_results[role]["selections_by_lane"],
                "threshold_distances_by_lane": role_results[role]["threshold_distances_by_lane"],
            }, sort_keys=True), flush=True)

        evidence_components = {
            path.relative_to(evidence_root).as_posix(): "sha256:" + base.sha_file(path)
            for path in sorted(evidence_root.rglob("*")) if path.is_file() and path.name != "evidence_manifest.json"
        }
        evidence_manifest = {
            "manifest_version": "diagnostic-scaling-recovery-evidence.v1",
            "run_id": run_id,
            "model_id": model_id,
            "revision": candidate["revision"],
            "protocol_sha256": contract_sha,
            "training_dataset_sha256": dataset_sha,
            "components": evidence_components,
            "component_count": len(evidence_components),
            "bytes": sum((evidence_root / name).stat().st_size for name in evidence_components),
            "cross_lane_ranking_performed": False,
            "promotion_performed": False,
            "lineage_mutated": False,
        }
        base.write_once_json(evidence_root / "evidence_manifest.json", evidence_manifest)
        export_prefix = f"{SCIENTIFIC_PREFIX}/diagnostic_scaling_recovery/{run_id}/models/{base.slug(model_id)}"
        exported = base.upload_tree_verified(client, evidence_root, export_prefix)
        manifest_key = f"{export_prefix}/evidence_manifest.json"
        manifest_record = next(row for row in exported if row["key"] == manifest_key)

        result.update({
            "status": "completed",
            "disposition": "scientific_diagnostic_scaling_recovery_model_complete",
            "training_performed": True,
            "training_dataset_sha256": dataset_sha,
            "roles": role_results,
            "evidence_prefix": export_prefix,
            "evidence_manifest": manifest_record,
            "exported_file_count": len(exported),
            "exported_bytes": sum(int(row["bytes"]) for row in exported),
            "cross_lane_ranking_performed": False,
            "promotion_performed": False,
            "lineage_mutated": False,
        })
        return 0
    except Exception as exc:
        result.update({
            "status": "failed",
            "disposition": "runtime_or_evidence_failure",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "promotion_performed": False,
            "lineage_mutated": False,
        })
        raise
    finally:
        result["completed_at_unix"] = time.time()
        terminal_raw = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        client.put_object(Bucket=base.bucket(), Key=terminal_key, Body=terminal_raw)
        observed = base.read_s3(client, terminal_key)
        if observed != terminal_raw:
            raise RuntimeError("terminal Diagnostic Scaling Recovery S3 readback mismatch")
        print("DIAGNOSTIC_SCALING_RECOVERY_TERMINAL_JSON " + json.dumps({
            "key": terminal_key,
            "sha256": sha(terminal_raw),
            "bytes": len(terminal_raw),
            "status": result["status"],
        }, sort_keys=True), flush=True)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
