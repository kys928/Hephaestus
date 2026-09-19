#!/usr/bin/env python3
"""Run the bounded GLM tiny-adaptation probe.

Requires a passed cheap-screen result. Trains exactly three diagnosis-role LoRA
optimizer steps on the original governed Test-3 bytes, checkpoints every training
step to S3, then reruns the five cheap-screen probes. This is adaptation screening,
not certification or promotion.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preflight_diagnostic_scaling_recovery_v1 as recovery_preflight
import run_adaptation_elasticity_v1 as elastic
import run_cognitive_topology_v1 as topology_runner
import run_diagnostic_scaling_v1 as base
import run_glm_cheap_screen_v1 as cheap
import launch_first_bounded_scientific_training as storage

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/experiments/hephaestus_glm_tiny_adaptation_v1.json"
SCREEN_PATH = ROOT / "configs/experiments/hephaestus_glm_cheap_screen_v1.json"
RECOVERY_PATH = ROOT / "configs/experiments/hephaestus_diagnostic_scaling_recovery_v1.json"
SEMANTIC_PATH = ROOT / "configs/eval_packs/semantic_behavior_v1.yaml"
TOPOLOGY_PATH = ROOT / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"
WORK_ROOT = Path("/opt/hephaestus-glm-tiny-adaptation")


def required(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_prerequisite(result: dict[str, Any], protocol: dict[str, Any]) -> None:
    req = protocol["prerequisite"]
    checks = {
        "result_version": result.get("result_version") == req["required_result_version"],
        "status": result.get("status") == req["required_status"],
        "advance": result.get("advance_to_tiny_adaptation") is bool(req["required_advance_to_tiny_adaptation"]),
        "screening_only": result.get("screening_only") is bool(req["require_screening_only"]),
        "no_training": result.get("training_performed") is not bool(req["require_training_performed_false"]),
        "model": result.get("model_id") == protocol["model"]["model_id"],
        "revision": result.get("revision") == protocol["model"]["revision"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("cheap-screen prerequisite failed: " + ",".join(failed))


def heartbeat(client: Any, prefix: str, run_id: str, stage: str, *, deadline: float, **extra: Any) -> None:
    payload = {
        "heartbeat_version": "hephaestus-glm-tiny-adaptation-heartbeat.v1",
        "run_id": run_id,
        "stage": stage,
        "timestamp_unix": time.time(),
        "remaining_internal_wall_seconds": max(0.0, deadline - time.monotonic()),
        **extra,
    }
    cheap.put_json_verified(client, f"{prefix}/heartbeats/{int(time.time() * 1000)}-{stage}.json", payload)
    cheap.put_json_verified(client, f"{prefix}/progress.json", payload)
    print("GLM_TINY_ADAPTATION_HEARTBEAT_JSON " + json.dumps(payload, sort_keys=True), flush=True)


def training_rows(client: Any, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    training_key, contamination_key = recovery_preflight.source_keys()
    training_raw = storage.read_key(client, training_key)
    contamination_raw = storage.read_key(client, contamination_key)
    expected_sha = protocol["training"]["source_training_dataset_sha256"]
    if sha(training_raw) != expected_sha:
        raise RuntimeError("tiny-adaptation source training SHA drifted")
    contamination = json.loads(contamination_raw)
    if contamination.get("passed") is not True:
        raise RuntimeError("tiny-adaptation source contamination evidence does not pass")
    rows = [json.loads(line) for line in training_raw.decode("utf-8").splitlines() if line.strip()]
    role = protocol["training"]["role"]
    rows = [row for row in rows if row.get("role") == role]
    required_count = int(protocol["training"]["training_examples_consumed"])
    if len(rows) < required_count:
        raise RuntimeError(f"insufficient {role} training rows: {len(rows)} < {required_count}")
    rng = random.Random(int(protocol["training"]["shuffle_seed"]))
    rng.shuffle(rows)
    return rows[:required_count], training_key, contamination_key


def build_peft(snapshot: Path, candidate: dict[str, Any], recovery: dict[str, Any], protocol: dict[str, Any]):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    model, tokenizer, runtime = elastic._load_training_base(snapshot, candidate, recovery)
    props = torch.cuda.get_device_properties(0)
    if props.total_memory / (1024 ** 3) + 1e-9 < float(protocol["execution"]["minimum_gpu_memory_gib"]):
        raise RuntimeError("tiny-adaptation GPU is below configured memory floor")
    target_modules, target_parameters, rank_pattern = base.target_surface(model, candidate, recovery)
    train = protocol["training"]
    kwargs: dict[str, Any] = {
        "r": int(train["rank"]),
        "lora_alpha": int(train["alpha"]),
        "lora_dropout": float(train["dropout"]),
        "bias": str(train["bias"]),
        "task_type": TaskType.CAUSAL_LM,
        "target_modules": target_modules,
    }
    if target_parameters:
        kwargs["target_parameters"] = target_parameters
        kwargs["rank_pattern"] = rank_pattern
    peft_model = get_peft_model(model, LoraConfig(**kwargs))
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    runtime.update({
        "tiny_adaptation": True,
        "gpu": props.name,
        "gpu_memory_bytes": int(props.total_memory),
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "target_module_count": len(target_modules),
        "target_parameter_count": len(target_parameters),
        "training_optimizer_steps": int(train["optimizer_steps"]),
    })
    return peft_model, tokenizer, trainable, runtime, target_modules, target_parameters


def train_three_steps(
    model: Any,
    tokenizer: Any,
    trainable: list[Any],
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    client: Any,
    prefix: str,
    run_id: str,
    deadline: float,
) -> list[dict[str, Any]]:
    import torch

    train = protocol["training"]
    steps = int(train["optimizer_steps"])
    accum = int(train["gradient_accumulation_steps"])
    if steps != 3 or accum * steps != len(rows):
        raise RuntimeError("tiny-adaptation training geometry drifted")
    torch.manual_seed(int(train["model_seed"]))
    torch.cuda.manual_seed_all(int(train["model_seed"]))
    tokenized = [elastic._tokenize_example(tokenizer, row, int(train["max_seq_length"])) for row in rows]
    optimizer = torch.optim.AdamW(trainable, lr=float(train["learning_rate"]), weight_decay=float(train["weight_decay"]))
    model.train()
    step_rows: list[dict[str, Any]] = []
    cursor = 0
    for optimizer_step in range(1, steps + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError("tiny-adaptation deadline reached before optimizer step")
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        started = time.perf_counter()
        for _ in range(accum):
            if time.monotonic() >= deadline:
                raise TimeoutError("tiny-adaptation deadline reached during gradient accumulation")
            batch = elastic._tensorize(tokenized[cursor])
            cursor += 1
            out = model(**batch, use_cache=False)
            raw_loss = out.loss
            if not bool(torch.isfinite(raw_loss)):
                raise RuntimeError("non-finite tiny-adaptation loss")
            losses.append(float(raw_loss.detach().item()))
            (raw_loss / accum).backward()
            del out, raw_loss, batch
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(train["max_grad_norm"])).item())
        optimizer.step()
        torch.cuda.synchronize()
        payload = {
            "training_step_version": "hephaestus-glm-tiny-adaptation-step.v1",
            "optimizer_step": optimizer_step,
            "micro_batches": accum,
            "mean_loss": sum(losses) / len(losses),
            "min_loss": min(losses),
            "max_loss": max(losses),
            "grad_norm": grad_norm,
            "seconds": time.perf_counter() - started,
            "allocated_vram_bytes": int(torch.cuda.memory_allocated()),
            "reserved_vram_bytes": int(torch.cuda.memory_reserved()),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        }
        step_rows.append(payload)
        cheap.put_json_verified(client, f"{prefix}/training/step_{optimizer_step:03d}.json", payload)
        heartbeat(client, prefix, run_id, "training_step_complete", deadline=deadline, optimizer_step=optimizer_step, mean_loss=payload["mean_loss"])
    del optimizer
    return step_rows


def evaluate_post(
    model: Any,
    tokenizer: Any,
    candidate: dict[str, Any],
    screen: dict[str, Any],
    semantic: dict[str, Any],
    topology: dict[str, Any],
    client: Any,
    prefix: str,
    run_id: str,
    deadline: float,
) -> list[dict[str, Any]]:
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    rows: list[dict[str, Any]] = []
    seed = int(screen["generation"]["seed"])
    for index, probe in enumerate(screen["probes"], start=1):
        if time.monotonic() >= deadline:
            raise TimeoutError("tiny-adaptation deadline reached before post-adaptation probe")
        if probe["source"] == "semantic":
            prompt = str(cheap.semantic_task(semantic, str(probe["task_id"]))["prompt"])
        else:
            case = cheap.topology_case(topology, str(probe["case_id"]))
            prompt = topology_runner._prompt(topology, case)
        heartbeat(client, prefix, run_id, "post_probe_started", deadline=deadline, probe_id=probe["probe_id"], probe_index=index)
        generated = cheap.generate_one(
            model,
            tokenizer,
            prompt,
            candidate=candidate,
            lane=str(probe["lane"]),
            seed=seed,
            max_new_tokens=int(probe["max_new_tokens"]),
            deadline_monotonic=deadline,
        )
        raw_output = str(generated.pop("output"))
        projected = cheap.project_reasoning(raw_output, str(probe["lane"]))
        score, probe_pass = cheap.score_probe(probe, output=projected, semantic=semantic, topology=topology, seed=seed)
        if probe["lane"] == "reasoning_aware" and generated["finish_reason"] in {"max_tokens", "runtime_deadline"}:
            probe_pass = None
        row = {
            "sample_version": "hephaestus-glm-tiny-adaptation-post-sample.v1",
            "probe_index": index,
            "probe_id": probe["probe_id"],
            "lane": probe["lane"],
            "seed": seed,
            "raw_output": raw_output,
            "output": projected,
            "generation": generated,
            "score": score,
            "probe_pass": probe_pass,
        }
        rows.append(row)
        cheap.put_json_verified(client, f"{prefix}/post_adaptation_samples/{index:03d}-{probe['probe_id']}.json", row)
        heartbeat(client, prefix, run_id, "post_probe_complete", deadline=deadline, probe_id=probe["probe_id"], probe_pass=probe_pass)
        if generated["runtime_deadline_hit"]:
            break
    return rows


def disposition(step_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[str, bool]:
    gate = protocol["advance_gate"]
    if len(step_rows) != int(protocol["training"]["optimizer_steps"]):
        return "tiny_adaptation_inconclusive_training_incomplete", False
    losses = [float(row["mean_loss"]) for row in step_rows]
    if not losses or not all(math.isfinite(value) for value in losses):
        return "tiny_adaptation_failed_non_finite_loss", False
    if losses[-1] > losses[0] * float(gate["maximum_final_to_first_step_loss_ratio"]):
        return "tiny_adaptation_not_promising_loss_instability", False
    expected = int(protocol["post_adaptation_probes"]["required_probe_count"])
    if len(probe_rows) != expected:
        return "tiny_adaptation_inconclusive_post_probe_incomplete", False
    if any(row["generation"]["runtime_deadline_hit"] for row in probe_rows):
        return "tiny_adaptation_inconclusive_runtime_deadline", False
    if any(row["lane"] == "reasoning_aware" and row["generation"]["finish_reason"] == "max_tokens" for row in probe_rows):
        return "tiny_adaptation_inconclusive_reasoning_budget_exhausted", False
    if not all(row.get("probe_pass") is True for row in probe_rows):
        return "tiny_adaptation_not_promising_behavioral_regression", False
    return "tiny_adaptation_passed_eligible_for_full_recovery_consideration_only", True


def main() -> int:
    import torch

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    screen = json.loads(SCREEN_PATH.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    semantic = json.loads(SEMANTIC_PATH.read_text(encoding="utf-8"))
    topology = json.loads(TOPOLOGY_PATH.read_text(encoding="utf-8"))
    model_id = protocol["model"]["model_id"]
    candidate = next(row for row in recovery["candidates"] if row["model_id"] == model_id)
    for key in ("revision", "model_type", "loader"):
        if candidate[key] != protocol["model"][key]:
            raise RuntimeError(f"tiny-adaptation model identity drift: {key}")

    run_id = required("HEPHAESTUS_GLM_TINY_ADAPT_RUN_ID")
    repo_sha = required("HEPHAESTUS_REPO_SHA")
    cheap_result_key = required("HEPHAESTUS_GLM_SCREEN_RESULT_KEY")
    client = base.s3_client()
    cheap_result = json.loads(base.read_s3(client, cheap_result_key))
    validate_prerequisite(cheap_result, protocol)

    prefix = f"{protocol['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    deadline = time.monotonic() + int(protocol["execution"]["hard_wall_seconds"])
    model = tokenizer = None
    step_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "result_version": "hephaestus-glm-tiny-adaptation.v1",
        "protocol_id": protocol["protocol_id"],
        "status": "running",
        "screening_only": True,
        "adaptation_probe_only": True,
        "certification_claim_allowed": False,
        "training_performed": True,
        "promotion_performed": False,
        "lineage_mutated": False,
        "run_id": run_id,
        "repo_sha": repo_sha,
        "model_id": model_id,
        "revision": candidate["revision"],
        "cheap_screen_result_key": cheap_result_key,
        "s3_prefix": prefix,
        "started_at_unix": time.time(),
    }
    adapter_dir = WORK_ROOT / run_id / "adapter"
    try:
        cheap.put_json_verified(client, f"{prefix}/run_manifest.json", {
            "protocol": protocol,
            "repo_sha": repo_sha,
            "cheap_screen_result_key": cheap_result_key,
            "training_performed": True,
            "full_recovery_performed": False,
        })
        heartbeat(client, prefix, run_id, "prerequisite_verified", deadline=deadline)
        rows, training_key, contamination_key = training_rows(client, protocol)
        cheap.put_json_verified(client, f"{prefix}/training_source.json", {
            "training_key": training_key,
            "contamination_key": contamination_key,
            "dataset_sha256": protocol["training"]["source_training_dataset_sha256"],
            "selected_role": protocol["training"]["role"],
            "selected_examples": len(rows),
        })
        heartbeat(client, prefix, run_id, "materializing_model", deadline=deadline)
        snapshot, manifest = base.materialize_model(candidate, WORK_ROOT / run_id / "model")
        cheap.put_json_verified(client, f"{prefix}/model_manifest.json", manifest)
        heartbeat(client, prefix, run_id, "loading_training_model", deadline=deadline)
        model, tokenizer, trainable, runtime, target_modules, target_parameters = build_peft(snapshot, candidate, recovery, protocol)
        cheap.put_json_verified(client, f"{prefix}/runtime.json", runtime)
        cheap.put_json_verified(client, f"{prefix}/target_surface.json", {
            "target_modules": target_modules,
            "target_parameters": target_parameters,
            "trainable_parameters": runtime["trainable_parameters"],
        })
        heartbeat(client, prefix, run_id, "training_model_loaded", deadline=deadline, trainable_parameters=runtime["trainable_parameters"])

        torch.cuda.reset_peak_memory_stats()
        step_rows = train_three_steps(model, tokenizer, trainable, rows, protocol, client, prefix, run_id, deadline)
        heartbeat(client, prefix, run_id, "training_complete", deadline=deadline, optimizer_steps=len(step_rows))

        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir, safe_serialization=True)
        uploaded = base.upload_tree_verified(client, adapter_dir, f"{prefix}/final_adapter")
        cheap.put_json_verified(client, f"{prefix}/final_adapter_manifest.json", {
            "adapter_files": len(uploaded),
            "adapter_bytes": sum(int(row["bytes"]) for row in uploaded),
            "optimizer_steps": len(step_rows),
            "promotion_performed": False,
            "lineage_mutated": False,
        })
        heartbeat(client, prefix, run_id, "final_adapter_preserved", deadline=deadline)

        probe_rows = evaluate_post(model, tokenizer, candidate, screen, semantic, topology, client, prefix, run_id, deadline)
        disp, advance = disposition(step_rows, probe_rows, protocol)
        result.update({
            "status": "completed",
            "disposition": disp,
            "advance_to_full_recovery": advance,
            "full_recovery_performed": False,
            "optimizer_steps_completed": len(step_rows),
            "post_adaptation_probes_completed": len(probe_rows),
            "training_step_losses": [row["mean_loss"] for row in step_rows],
            "post_adaptation_probe_results": [
                {
                    "probe_id": row["probe_id"],
                    "lane": row["lane"],
                    "probe_pass": row["probe_pass"],
                    "finish_reason": row["generation"]["finish_reason"],
                    "generated_tokens": row["generation"]["generated_tokens"],
                }
                for row in probe_rows
            ],
        })
        print("GLM_TINY_ADAPTATION_RESULT_JSON " + json.dumps({
            "status": result["status"],
            "disposition": disp,
            "advance_to_full_recovery": advance,
            "optimizer_steps_completed": len(step_rows),
            "post_adaptation_probes_completed": len(probe_rows),
        }, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        result.update({
            "status": "failed",
            "disposition": "tiny_adaptation_runtime_or_evidence_failure",
            "advance_to_full_recovery": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "optimizer_steps_completed": len(step_rows),
            "post_adaptation_probes_completed": len(probe_rows),
        })
        raise
    finally:
        result["completed_at_unix"] = time.time()
        cheap.put_json_verified(client, f"{prefix}/result.json", result)
        try:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())