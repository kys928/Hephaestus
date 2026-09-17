#!/usr/bin/env python3
"""Run one Diagnostic Scaling V1 candidate entirely on ephemeral Pod storage.

The driver downloads the immutable launch admission and model snapshot, evaluates
Diagnosis and Controller at dose zero, trains the frozen LoRA dose trajectory,
exports compact evidence plus selected adapters to RunPod S3, verifies every
exported byte, and writes the terminal execution record last.

No Network Volume mount is used and no promotion or lineage mutation is allowed.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
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
from hephaestus.scoring.behavioral import evaluate_behavioral_sample

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/experiments/hephaestus_diagnostic_scaling_v1.json"
TOPOLOGY_PATH = ROOT / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"
SEMANTIC_PATH = ROOT / "configs/eval_packs/semantic_behavior_v1.yaml"
SCIENTIFIC_PREFIX = "hephaestus/scientific/v1"
EPHEMERAL_ROOT = Path("/opt/hephaestus-diagnostic-scaling")


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_once_json(path: Path, payload: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite scientific evidence: {path}")
    atomic_json(path, payload)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "--" for ch in value).strip("-.")


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def s3_client() -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=required("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),
        region_name=required("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(retries={"mode": "standard", "max_attempts": 10}),
    )


def bucket() -> str:
    # This is the Network Volume's S3 bucket identity only. The Pod itself has no
    # Network Volume attachment; the launcher contract forbids one.
    return required("RUNPOD_NETWORK_VOLUME_ID")


def read_s3(client: Any, key: str) -> bytes:
    response = client.get_object(Bucket=bucket(), Key=key)
    body = response["Body"]
    try:
        return body.read()
    finally:
        body.close()


def download_s3(client: Any, key: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket(), key, str(target))


def verify_s3_file(client: Any, key: str, source: Path) -> dict[str, Any]:
    expected = sha_file(source)
    response = client.get_object(Bucket=bucket(), Key=key)
    body = response["Body"]
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            chunk = body.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    finally:
        body.close()
    observed = digest.hexdigest()
    if observed != expected or size != source.stat().st_size:
        raise RuntimeError(f"S3 readback mismatch for {key}")
    return {"key": key, "sha256": expected, "bytes": size}


def upload_tree_verified(client: Any, root: Path, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        key = f"{prefix}/{relative}"
        client.upload_file(str(path), bucket(), key)
        rows.append(verify_s3_file(client, key, path))
    return rows


def project_reasoning(raw: str, contract: dict[str, Any]) -> str:
    projection = contract["evaluation"]["reasoning_projection"]
    if projection.get("score_projection") != "text_after_last_reasoning_close_delimiter_else_raw":
        raise RuntimeError("unsupported reasoning projection")
    best_end = -1
    for delimiter in projection["reasoning_close_delimiters"]:
        index = raw.rfind(str(delimiter))
        if index >= 0:
            best_end = max(best_end, index + len(str(delimiter)))
    return (raw[best_end:] if best_end >= 0 else raw).strip()


def semantic_tasks(semantic: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for key in ("generation_probes", "continuation_prompts", "structure_tests", "repetition_checks", "length_termination_checks"):
        tasks.extend(dict(row) for row in semantic.get(key, []))
    expected = ["instruction_triplet", "planet_fact", "observatory_continuation", "structured_planet_answer", "anti_repetition", "brief_termination"]
    if [str(row.get("task_id")) for row in tasks] != expected:
        raise RuntimeError("semantic pack task identity/order drifted")
    return tasks


def evaluate_role(
    model: Any,
    tokenizer: Any,
    topology: dict[str, Any],
    contract: dict[str, Any],
    candidate: dict[str, Any],
    role: str,
    seeds: list[int],
    sample_path: Path,
) -> dict[str, Any]:
    role_cases = [case for case in topology["cases"] if case["role"] == role]
    role_spec = dict(topology)
    role_spec["roles"] = [role]
    role_spec["generation"] = dict(topology["generation"])
    role_spec["generation"]["seeds"] = list(seeds)
    role_spec["generation"]["repetitions"] = len(seeds)
    samples: list[dict[str, Any]] = []
    tokenizer.padding_side = "left"
    model.eval()
    for seed in seeds:
        for case in role_cases:
            prompt = topology_runner._prompt(topology, case)
            generated = topology_runner._generate_one(
                model, tokenizer, prompt, seed=int(seed),
                max_new_tokens=int(topology["generation"]["max_new_tokens"]),
            )
            raw_output = generated["output"]
            scored_output = project_reasoning(raw_output, contract)
            score = topology_runner.score_response(topology, case, scored_output)
            row = {
                "model_id": candidate["model_id"],
                "revision": candidate["revision"],
                "case_id": case["case_id"],
                "role": role,
                "condition": case["condition"],
                "pair_id": case.get("pair_id"),
                "seed": int(seed),
                "output": scored_output,
                "raw_output": raw_output,
                "reasoning_projection_applied": scored_output != raw_output.strip(),
                "generation": {key: value for key, value in generated.items() if key != "output"},
                "score": score,
            }
            samples.append(row)
            append_jsonl(sample_path, row)
    expected = len(role_cases) * len(seeds)
    if len(samples) != expected:
        raise RuntimeError(f"incomplete role screen: {len(samples)} != {expected}")
    summary = topology_runner._summarize_model(role_spec, candidate, samples, {})["roles"][role]
    tokenizer.padding_side = "right"
    return {"summary": summary, "samples": samples}


def evaluate_semantic(
    model: Any,
    tokenizer: Any,
    semantic: dict[str, Any],
    contract: dict[str, Any],
    candidate: dict[str, Any],
    seeds: list[int],
    sample_path: Path,
) -> dict[str, Any]:
    tasks = semantic_tasks(semantic)
    rows: list[dict[str, Any]] = []
    hard_failures: set[str] = set()
    tokenizer.padding_side = "left"
    model.eval()
    for seed in seeds:
        for task in tasks:
            generated = topology_runner._generate_one(
                model, tokenizer, str(task["prompt"]), seed=int(seed),
                max_new_tokens=int(semantic["decoding_config"]["max_new_tokens"]),
            )
            raw_output = generated["output"]
            scored_output = project_reasoning(raw_output, contract)
            score = evaluate_behavioral_sample(task, scored_output, seed)
            for check in score.checks:
                if check.hard and not check.passed:
                    hard_failures.add(f"{task['task_id']}:{seed}:{check.name}")
            row = {
                "model_id": candidate["model_id"],
                "revision": candidate["revision"],
                "task_id": task["task_id"],
                "seed": int(seed),
                "output": scored_output,
                "raw_output": raw_output,
                "reasoning_projection_applied": scored_output != raw_output.strip(),
                "generation": {key: value for key, value in generated.items() if key != "output"},
                "score": score.to_dict(),
            }
            rows.append(row)
            append_jsonl(sample_path, row)
    expected = len(tasks) * len(seeds)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete semantic evidence: {len(rows)} != {expected}")
    summary = {
        "sample_count": len(rows),
        "task_count": len(tasks),
        "mean_score": mean([float(row["score"]["overall_score"]) for row in rows]),
        "deterministic_pass_rate": mean([float(row["score"]["deterministic_passed"]) for row in rows]),
        "hard_failures": sorted(hard_failures),
        "hard_failure_count": len(hard_failures),
        "reasoning_projection_count": sum(int(row["reasoning_projection_applied"]) for row in rows),
    }
    tokenizer.padding_side = "right"
    return {"summary": summary, "samples": rows}


def safe_state(
    role_summary: dict[str, Any],
    semantic_summary: dict[str, Any],
    baseline_semantic: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    base_failures = set(baseline_semantic["hard_failures"])
    adapted_failures = set(semantic_summary["hard_failures"])
    new_hard = sorted(adapted_failures - base_failures)
    semantic_delta = float(semantic_summary["mean_score"]) - float(baseline_semantic["mean_score"])
    evaluation = contract["evaluation"]
    interface_ok = (
        float(role_summary["schema_compliance"]) >= float(evaluation["schema_compliance_required"])
        and float(role_summary["hallucination_rate"]) <= float(evaluation["hallucination_rate_required"])
    )
    strict_safe = interface_ok and not new_hard and semantic_delta >= float(evaluation["strict_semantic_delta_min"]) - 1e-12
    bounded_safe = interface_ok and not new_hard and semantic_delta >= float(evaluation["bounded_semantic_delta_min"]) - 1e-12
    return {
        "interface_ok": interface_ok,
        "new_hard_failures": new_hard,
        "semantic_score_delta": semantic_delta,
        "strict_safe": strict_safe,
        "bounded_safe": bounded_safe,
    }


def select_points(points: list[dict[str, Any]]) -> dict[str, int | None]:
    def best(rows: list[dict[str, Any]]) -> int | None:
        if not rows:
            return None
        winner = max(rows, key=lambda row: (float(row["role"]["quality_100"]), -int(row["optimizer_steps"])))
        return int(winner["optimizer_steps"])

    return {
        "raw_peak_steps": best([row for row in points if row["safety"]["interface_ok"]]),
        "strict_safe_peak_steps": best([row for row in points if row["safety"]["strict_safe"]]),
        "bounded_safe_peak_steps": best([row for row in points if row["safety"]["bounded_safe"]]),
    }


def threshold_distances(points: list[dict[str, Any]], thresholds: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"raw": {}, "strict_safe": {}, "bounded_safe": {}}
    for threshold in thresholds:
        for mode in out:
            match = None
            for point in sorted(points, key=lambda row: int(row["optimizer_steps"])):
                if float(point["role"]["quality_100"]) + 1e-12 < threshold:
                    continue
                if not point["safety"]["interface_ok"]:
                    continue
                if mode == "strict_safe" and not point["safety"]["strict_safe"]:
                    continue
                if mode == "bounded_safe" and not point["safety"]["bounded_safe"]:
                    continue
                match = {
                    "optimizer_steps": int(point["optimizer_steps"]),
                    "approx_epochs": float(point["approx_epochs"]),
                    "supervised_tokens": int(point["training"]["supervised_tokens_cumulative"]),
                    "quality_100": float(point["role"]["quality_100"]),
                    "semantic_score_delta": float(point["safety"]["semantic_score_delta"]),
                }
                break
            out[mode][str(threshold)] = match
    return out


def unload(model: Any | None = None, tokenizer: Any | None = None) -> None:
    del model, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def load_base(snapshot: Path, candidate: dict[str, Any], contract: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    if not cuda_available or device_count != 1:
        diagnostic = {
            "cuda_available": cuda_available,
            "device_count": device_count,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        }
        try:
            import subprocess
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader"],
                check=False, capture_output=True, text=True, timeout=20,
            )
            diagnostic["nvidia_smi_returncode"] = probe.returncode
            diagnostic["nvidia_smi_stdout"] = probe.stdout.strip()
            diagnostic["nvidia_smi_stderr"] = probe.stderr.strip()
        except Exception as exc:
            diagnostic["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            "Diagnostic Scaling V1 requires exactly one visible CUDA GPU; observed="
            + json.dumps(diagnostic, sort_keys=True)
        )
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    memory_gib = props.total_memory / (1024 ** 3)
    if memory_gib + 1e-9 < float(candidate["minimum_gpu_memory_gib"]):
        raise RuntimeError(
            f"allocated GPU is below candidate memory floor: {props.name} {memory_gib:.2f} GiB < {candidate['minimum_gpu_memory_gib']}"
        )
    model, tokenizer, runtime = elastic._load_training_base(snapshot, candidate, contract)
    runtime["contract_minimum_gpu_memory_gib"] = candidate["minimum_gpu_memory_gib"]
    runtime["observed_gpu_memory_gib"] = memory_gib
    return model, tokenizer, runtime


def target_surface(model: Any, candidate: dict[str, Any], contract: dict[str, Any]) -> tuple[list[str], list[str], dict[str, int]]:
    import torch

    policy = contract["training"]["target_policy"][candidate["model_type"]]
    module_suffixes = set(policy["target_modules"])
    modules = sorted(
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in module_suffixes
    )
    if not modules:
        raise RuntimeError(f"no governed LoRA target modules found for {candidate['model_id']}")
    found_suffixes = {name.rsplit(".", 1)[-1] for name in modules}
    missing_suffixes = module_suffixes - found_suffixes
    if missing_suffixes:
        raise RuntimeError(f"governed target module suffixes absent for {candidate['model_id']}: {sorted(missing_suffixes)}")

    parameter_suffixes = list(policy["target_parameters"])
    parameters: list[str] = []
    if parameter_suffixes:
        for name, parameter in model.named_parameters():
            if any(name.endswith(suffix) for suffix in parameter_suffixes):
                if parameter.ndim not in (2, 3):
                    raise RuntimeError(f"unsupported governed target parameter rank: {name} ndim={parameter.ndim}")
                parameters.append(name)
        parameters = sorted(parameters)
        for suffix in parameter_suffixes:
            if not any(name.endswith(suffix) for name in parameters):
                raise RuntimeError(f"governed target parameter suffix absent for {candidate['model_id']}: {suffix}")
    rank_pattern = {name: int(policy["expert_rank"]) for name in parameters} if parameters else {}
    return modules, parameters, rank_pattern


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
    components = {
        path.relative_to(adapter_dir).as_posix(): "sha256:" + sha_file(path)
        for path in sorted(adapter_dir.rglob("*")) if path.is_file() and path.name != "hephaestus_diagnostic_scaling_adapter_manifest.json"
    }
    return {
        "manifest_version": "diagnostic-scaling-adapter.v1",
        "protocol_sha256": contract_sha,
        "training_dataset_sha256": dataset_sha,
        "model_id": candidate["model_id"],
        "revision": candidate["revision"],
        "role": role,
        "optimizer_steps": optimizer_steps,
        "trainable_parameters": trainable_parameters,
        "target_module_count": len(target_modules),
        "target_modules_sha256": sha("\n".join(target_modules).encode()),
        "target_parameter_count": len(target_parameters),
        "target_parameters_sha256": sha("\n".join(target_parameters).encode()),
        "components": components,
        "adapter_bytes": sum((adapter_dir / name).stat().st_size for name in components),
    }


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
    baseline_semantic: dict[str, Any],
    evidence_root: Path,
    dataset_sha: str,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import get_linear_schedule_with_warmup

    model = tokenizer = None
    role_root = evidence_root / "roles" / role
    role_root.mkdir(parents=True, exist_ok=False)
    try:
        model, tokenizer, runtime = load_base(snapshot, candidate, contract)
        topology_runner._warmup(model, tokenizer)
        target_modules, target_parameters, rank_pattern = target_surface(model, candidate, contract)
        atomic_json(role_root / "runtime.json", runtime)
        atomic_json(role_root / "target_surface.json", {
            "target_modules": target_modules,
            "target_parameters": target_parameters,
            "rank_pattern": rank_pattern,
        })

        screen_seed = [int(contract["evaluation"]["screening_seed"])]
        semantic_seeds = [int(v) for v in semantic["decoding_config"]["seeds"]]
        baseline_role = evaluate_role(
            model, tokenizer, topology, contract, candidate, role, screen_seed,
            role_root / "samples" / "role_step_000.jsonl",
        )
        zero_safety = safe_state(baseline_role["summary"], baseline_semantic, baseline_semantic, contract)
        points: list[dict[str, Any]] = [{
            "optimizer_steps": 0,
            "approx_epochs": 0.0,
            "role": baseline_role["summary"],
            "semantic": baseline_semantic,
            "safety": zero_safety,
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
                write_once_json(adapter_dir / "hephaestus_diagnostic_scaling_adapter_manifest.json", manifest)
                role_eval = evaluate_role(
                    peft_model, tokenizer, topology, contract, candidate, role, screen_seed,
                    role_root / "samples" / f"role_step_{optimizer_step:03d}.jsonl",
                )
                sem_eval = evaluate_semantic(
                    peft_model, tokenizer, semantic, contract, candidate, semantic_seeds,
                    role_root / "samples" / f"semantic_step_{optimizer_step:03d}.jsonl",
                )
                safety = safe_state(role_eval["summary"], sem_eval["summary"], baseline_semantic, contract)
                points.append({
                    "optimizer_steps": optimizer_step,
                    "approx_epochs": optimizer_step / float(train["expected_optimizer_steps_per_epoch"]),
                    "role": role_eval["summary"],
                    "semantic": sem_eval["summary"],
                    "safety": safety,
                    "training": {
                        "training_seconds_cumulative": cumulative_seconds,
                        "supervised_tokens_cumulative": cumulative_tokens,
                        "trainable_parameters": trainable_parameters,
                        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                        "mean_loss_since_previous_dose": mean(segment_losses),
                    },
                    "adapter": manifest,
                })
                segment_losses = []
                print("DIAGNOSTIC_SCALING_DOSE_JSON " + json.dumps({
                    "model_id": candidate["model_id"],
                    "role": role,
                    "optimizer_steps": optimizer_step,
                    "quality_100": role_eval["summary"]["quality_100"],
                    "strict_safe": safety["strict_safe"],
                    "bounded_safe": safety["bounded_safe"],
                    "semantic_score_delta": safety["semantic_score_delta"],
                }, sort_keys=True), flush=True)
                if optimizer_step >= max_steps:
                    stop = True
                    break
        if optimizer_step != max_steps:
            raise RuntimeError(f"LoRA trajectory stopped at {optimizer_step} != {max_steps}")

        selections = select_points(points)
        thresholds = threshold_distances(points, [int(value) for value in contract["evaluation"]["quality_thresholds_100"]])
        selected_steps = {int(step) for step in selections.values() if step is not None and int(step) > 0}
        for mode in thresholds.values():
            for row in mode.values():
                if row and int(row["optimizer_steps"]) > 0:
                    selected_steps.add(int(row["optimizer_steps"]))

        del optimizer, scheduler, outputs
        try:
            base_model = peft_model.unload()
            del peft_model, base_model
        except Exception:
            del peft_model
        unload(tokenizer=tokenizer)
        tokenizer = None

        confirmations: dict[str, Any] = {}
        confirmation_seeds = [int(value) for value in contract["evaluation"]["full_confirmation_seeds"]]
        confirmation_steps = sorted({int(step) for step in selections.values() if step is not None})
        for step in confirmation_steps:
            confirm_model = confirm_tokenizer = None
            try:
                confirm_model, confirm_tokenizer, _runtime = load_base(snapshot, candidate, contract)
                if step > 0:
                    confirm_model = PeftModel.from_pretrained(
                        confirm_model,
                        role_root / "adapters" / f"step_{step:03d}",
                        is_trainable=False,
                    )
                full = evaluate_role(
                    confirm_model, confirm_tokenizer, topology, contract, candidate, role, confirmation_seeds,
                    role_root / "confirmations" / f"role_step_{step:03d}.jsonl",
                )
                confirmations[str(step)] = full["summary"]
            finally:
                unload(confirm_model, confirm_tokenizer)

        # Keep only adapters that are scientifically selected or define a threshold crossing.
        adapters_root = role_root / "adapters"
        retained = []
        for adapter_dir in sorted(adapters_root.glob("step_*")):
            step = int(adapter_dir.name.split("_")[-1])
            if step in selected_steps:
                retained.append(step)
            else:
                shutil.rmtree(adapter_dir)

        curve = {
            "result_version": "diagnostic-scaling-role-curve.v1",
            "status": "complete",
            "protocol_sha256": contract_sha,
            "training_dataset_sha256": dataset_sha,
            "model_id": candidate["model_id"],
            "revision": candidate["revision"],
            "role": role,
            "points": points,
            "threshold_distances": thresholds,
            "selections": selections,
            "full_seed_confirmations": confirmations,
            "retained_adapter_steps": retained,
            "target_module_count": len(target_modules),
            "target_parameter_count": len(target_parameters),
            "trainable_parameters": trainable_parameters,
            "promotion_performed": False,
            "lineage_mutated": False,
        }
        write_once_json(role_root / "role_curve.json", curve)
        return curve
    finally:
        unload(model, tokenizer)


def verify_runtime_versions(admission_root: Path, contract: dict[str, Any]) -> dict[str, str]:
    exact = contract["execution"]["runtime_dependencies"]
    package_map = {
        "transformers": "transformers",
        "accelerate": "accelerate",
        "safetensors": "safetensors",
        "huggingface_hub": "huggingface-hub",
        "peft": "peft",
    }
    observed = {name: importlib.metadata.version(package) for name, package in package_map.items()}
    if observed != exact:
        raise RuntimeError(f"runtime dependency drift: {observed} != {exact}")
    requirements = (admission_root / "requirements.txt").read_text(encoding="utf-8")
    for name, version in exact.items():
        requirement_name = "huggingface-hub" if name == "huggingface_hub" else name
        if f"{requirement_name}=={version}" not in requirements.splitlines():
            raise RuntimeError(f"admission requirements missing exact runtime dependency: {requirement_name}=={version}")
    return observed


def materialize_model(candidate: dict[str, Any], model_root: Path) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(candidate["model_id"], revision=candidate["revision"], files_metadata=True)
    if str(info.sha) != candidate["revision"]:
        raise RuntimeError("immutable model revision drift during materialization")
    observed_license = str(getattr(getattr(info, "card_data", None), "license", "") or "").lower()
    if observed_license and observed_license != candidate["license"].lower():
        raise RuntimeError("model license drift during materialization")
    snapshot = Path(snapshot_download(
        repo_id=candidate["model_id"],
        revision=candidate["revision"],
        cache_dir=str(model_root / "hf_cache"),
    )).resolve()
    files = [path.relative_to(snapshot).as_posix() for path in sorted(snapshot.rglob("*")) if path.is_file()]
    if not any(name.endswith(".safetensors") for name in files):
        raise RuntimeError("immutable model materialization lacks safetensors")
    remote_weights = []
    for sibling in getattr(info, "siblings", []) or []:
        name = str(getattr(sibling, "rfilename", ""))
        if not name.endswith(".safetensors"):
            continue
        lfs = getattr(sibling, "lfs", None)
        remote_weights.append({
            "file": name,
            "sha256": getattr(lfs, "sha256", None) if lfs else None,
            "size": getattr(sibling, "size", None),
        })
    manifest = {
        "manifest_version": "diagnostic-scaling-model-snapshot.v1",
        "model_id": candidate["model_id"],
        "revision": candidate["revision"],
        "license": observed_license or candidate["license"].lower(),
        "trust_remote_code": False,
        "cache_persistence": "ephemeral",
        "snapshot_file_count": len(files),
        "snapshot_total_bytes": sum((snapshot / name).stat().st_size for name in files),
        "remote_safetensor_metadata": remote_weights,
    }
    return snapshot, manifest


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
        raise RuntimeError(f"model is not in frozen Diagnostic Scaling cohort: {model_id}")

    work_root = EPHEMERAL_ROOT / run_id / slug(model_id) / f"attempt-{attempt}"
    admission_root = work_root / "admission"
    model_root = work_root / "model"
    evidence_root = work_root / "evidence"
    if work_root.exists():
        shutil.rmtree(work_root)
    admission_root.mkdir(parents=True)
    evidence_root.mkdir(parents=True)

    client = s3_client()
    terminal_key = f"{SCIENTIFIC_PREFIX}/executions/{execution_id}/attempt-{attempt}/driver_result.json"
    result: dict[str, Any] = {
        "result_version": "diagnostic-scaling-driver.v1",
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
        admission_prefix = f"{SCIENTIFIC_PREFIX}/diagnostic_scaling/model_admission/{repo_sha}"
        for name in ("admission.json", "protocol.json", "training.jsonl", "contamination_report.json", "requirements.txt", "remote_models.json"):
            download_s3(client, f"{admission_prefix}/{name}", admission_root / name)
        admission = json.loads((admission_root / "admission.json").read_text(encoding="utf-8"))
        admitted_contract = (admission_root / "protocol.json").read_bytes()
        if admission.get("status") != "launch_inputs_admitted" or admission.get("repo_sha") != repo_sha:
            raise RuntimeError("Diagnostic Scaling launch admission identity mismatch")
        if admission.get("protocol_sha256") != contract_sha or admitted_contract != contract_raw:
            raise RuntimeError("runtime contract differs from admitted contract bytes")
        if admission.get("network_volume_attached") is not False or admission.get("promotion_allowed") is not False:
            raise RuntimeError("runtime admission violates ephemeral/governance boundary")
        if admission.get("candidate_revisions", {}).get(model_id) != candidate["revision"]:
            raise RuntimeError("candidate revision differs from launch admission")
        dataset_raw = (admission_root / "training.jsonl").read_bytes()
        dataset_sha = sha(dataset_raw)
        if dataset_sha != admission.get("training_dataset_sha256") or dataset_sha != contract["sources"]["source_training_dataset_sha256"]:
            raise RuntimeError("runtime training dataset differs from admitted source")
        runtime_versions = verify_runtime_versions(admission_root, contract)
        rows = [json.loads(line) for line in dataset_raw.decode("utf-8").splitlines() if line.strip()]
        roles = [*contract["roles"]["primary"], *contract["roles"]["calibration"]]
        per_role = {role: [row for row in rows if row.get("role") == role] for role in roles}
        expected = int(contract["training"]["examples_per_role"])
        if any(len(role_rows) != expected for role_rows in per_role.values()):
            raise RuntimeError("runtime role training data is incomplete")

        snapshot, model_manifest = materialize_model(candidate, model_root)
        write_once_json(evidence_root / "model_manifest.json", model_manifest)
        write_once_json(evidence_root / "run_manifest.json", {
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
            "network_volume_attached": False,
        })

        baseline_model = baseline_tokenizer = None
        try:
            baseline_model, baseline_tokenizer, baseline_runtime = load_base(snapshot, candidate, contract)
            topology_runner._warmup(baseline_model, baseline_tokenizer)
            baseline_semantic_eval = evaluate_semantic(
                baseline_model,
                baseline_tokenizer,
                semantic,
                contract,
                candidate,
                [int(value) for value in semantic["decoding_config"]["seeds"]],
                evidence_root / "baseline" / "semantic_samples.jsonl",
            )
            baseline_semantic = baseline_semantic_eval["summary"]
            write_once_json(evidence_root / "baseline" / "semantic_summary.json", baseline_semantic)
            write_once_json(evidence_root / "baseline" / "runtime.json", baseline_runtime)
        finally:
            unload(baseline_model, baseline_tokenizer)

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
                baseline_semantic=baseline_semantic,
                evidence_root=evidence_root,
                dataset_sha=dataset_sha,
            )
            print("DIAGNOSTIC_SCALING_ROLE_COMPLETE_JSON " + json.dumps({
                "model_id": model_id,
                "role": role,
                "selections": role_results[role]["selections"],
                "threshold_distances": role_results[role]["threshold_distances"],
            }, sort_keys=True), flush=True)

        evidence_components = {
            path.relative_to(evidence_root).as_posix(): "sha256:" + sha_file(path)
            for path in sorted(evidence_root.rglob("*")) if path.is_file() and path.name != "evidence_manifest.json"
        }
        evidence_manifest = {
            "manifest_version": "diagnostic-scaling-evidence.v1",
            "run_id": run_id,
            "model_id": model_id,
            "revision": candidate["revision"],
            "protocol_sha256": contract_sha,
            "training_dataset_sha256": dataset_sha,
            "components": evidence_components,
            "component_count": len(evidence_components),
            "bytes": sum((evidence_root / name).stat().st_size for name in evidence_components),
            "promotion_performed": False,
            "lineage_mutated": False,
        }
        write_once_json(evidence_root / "evidence_manifest.json", evidence_manifest)
        export_prefix = f"{SCIENTIFIC_PREFIX}/diagnostic_scaling/{run_id}/models/{slug(model_id)}"
        exported = upload_tree_verified(client, evidence_root, export_prefix)
        manifest_key = f"{export_prefix}/evidence_manifest.json"
        manifest_record = next(row for row in exported if row["key"] == manifest_key)

        result.update({
            "status": "completed",
            "disposition": "scientific_diagnostic_scaling_model_complete",
            "training_performed": True,
            "training_dataset_sha256": dataset_sha,
            "roles": role_results,
            "evidence_prefix": export_prefix,
            "evidence_manifest": manifest_record,
            "exported_file_count": len(exported),
            "exported_bytes": sum(int(row["bytes"]) for row in exported),
            "promotion_performed": False,
            "lineage_mutated": False,
        })
        print("DIAGNOSTIC_SCALING_MODEL_COMPLETE_JSON " + json.dumps({
            "run_id": run_id,
            "model_id": model_id,
            "revision": candidate["revision"],
            "evidence_manifest": manifest_record,
        }, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        result.update({
            "status": "failed",
            "disposition": "runtime_or_evidence_failure",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "training_performed": bool(result.get("training_performed")),
            "promotion_performed": False,
            "lineage_mutated": False,
        })
        raise
    finally:
        result["completed_at_unix"] = time.time()
        terminal_raw = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        client.put_object(Bucket=bucket(), Key=terminal_key, Body=terminal_raw)
        observed = read_s3(client, terminal_key)
        if observed != terminal_raw:
            raise RuntimeError("terminal Diagnostic Scaling S3 readback mismatch")
        print("DIAGNOSTIC_SCALING_TERMINAL_JSON " + json.dumps({
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
