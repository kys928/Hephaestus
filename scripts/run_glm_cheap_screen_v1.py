#!/usr/bin/env python3
"""Run the bounded GLM-4.7-Flash screening-only probe.

This protocol is intentionally not a certification run. It keeps the exact pinned
GLM revision and BF16 base weights, performs no training, evaluates one seed over
five small probes, writes every sample to governed S3 immediately, and stops under
a hard runtime deadline.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cognitive_topology_v1 as topology_runner
import run_diagnostic_scaling_v1 as base
from hephaestus.scoring.behavioral import evaluate_behavioral_sample

ROOT = Path(__file__).resolve().parents[1]
SCREEN_PATH = ROOT / "configs/experiments/hephaestus_glm_cheap_screen_v1.json"


def required(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def put_json_verified(client: Any, key: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    client.put_object(Bucket=base.bucket(), Key=key, Body=raw)
    observed = base.read_s3(client, key)
    if observed != raw:
        raise RuntimeError(f"S3 readback mismatch for {key}")
    return {"key": key, "bytes": len(raw)}


def project_reasoning(raw: str, lane: str) -> str:
    if lane == "fixed_non_thinking":
        return raw.strip()
    delimiter = "</think>"
    index = raw.rfind(delimiter)
    return (raw[index + len(delimiter):] if index >= 0 else raw).strip()


def load_bf16_base(snapshot: Path, candidate: dict[str, Any], screen: dict[str, Any]):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GLM cheap screen requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    memory_gib = props.total_memory / (1024 ** 3)
    minimum_gib = float(screen["execution"]["minimum_gpu_memory_gib"])
    if memory_gib + 1e-9 < minimum_gib:
        raise RuntimeError(f"cheap-screen GPU below {minimum_gib:g} GiB floor: {props.name} {memory_gib:.2f} GiB")

    cfg = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if cfg.model_type != candidate["model_type"]:
        raise RuntimeError(f"model type drifted: {cfg.model_type} != {candidate['model_type']}")
    if getattr(cfg, "quantization_config", None):
        raise RuntimeError("quantized GLM base is forbidden in cheap screen")

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    devices = {str(t.device) for t in (*model.parameters(), *model.buffers())}
    if not devices or not devices <= {"cuda", "cuda:0"}:
        raise RuntimeError(f"forbidden tensor residency in cheap screen: {sorted(devices)}")
    dtypes = {str(p.dtype) for p in model.parameters() if p.is_floating_point()}
    if dtypes != {"torch.bfloat16"}:
        raise RuntimeError(f"GLM cheap-screen base is not uniformly BF16: {sorted(dtypes)}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.config.use_cache = True

    return model, tokenizer, {
        "gpu": props.name,
        "gpu_memory_bytes": int(props.total_memory),
        "observed_gpu_memory_gib": memory_gib,
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "tensor_devices": sorted(devices),
        "weight_dtypes": sorted(dtypes),
        "quantization": False,
        "cpu_disk_offload": False,
        "use_kv_cache": True,
        "training_performed": False,
        "allocated_memory_bytes_after_load": int(torch.cuda.memory_allocated()),
    }


def normalized_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        return sorted({int(item) for item in value})
    try:
        return sorted({int(item) for item in value.tolist()})
    except Exception as exc:
        raise RuntimeError(f"unsupported EOS token id container: {type(value).__name__}") from exc


def generation_stop_token_ids(model: Any, tokenizer: Any, screen: dict[str, Any]) -> list[int]:
    model_generation_ids = normalized_token_ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
    model_config_ids = normalized_token_ids(getattr(getattr(model, "config", None), "eos_token_id", None))
    tokenizer_ids = normalized_token_ids(getattr(tokenizer, "eos_token_id", None))
    stop_ids = sorted(set(model_generation_ids) | set(model_config_ids) | set(tokenizer_ids))
    expected = sorted({int(item) for item in screen["generation"].get("expected_eos_token_ids", [])})
    if expected and stop_ids != expected:
        raise RuntimeError(
            f"GLM EOS token drift: observed={stop_ids} expected={expected}; "
            f"generation_config={model_generation_ids} model_config={model_config_ids} tokenizer={tokenizer_ids}"
        )
    if not stop_ids:
        raise RuntimeError("GLM generation has no EOS/turn-termination token ids")
    return stop_ids


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    candidate: dict[str, Any],
    lane: str,
    seed: int,
    max_new_tokens: int,
    deadline_monotonic: float,
    screen: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class RuntimeStopper(StoppingCriteria):
        def __init__(self) -> None:
            self.first_token_time: float | None = None
            self.deadline_hit = False

        def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
            now = time.monotonic()
            if self.first_token_time is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.first_token_time = time.perf_counter()
            if now >= deadline_monotonic:
                self.deadline_hit = True
                return True
            return False

    lane_cfg = candidate[lane]
    stop_token_ids = generation_stop_token_ids(model, tokenizer, screen)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **dict(lane_cfg.get("chat_template_kwargs") or {}),
    )
    encoded = tokenizer(rendered, return_tensors="pt", truncation=False)
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[1])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    clock = RuntimeStopper()
    started = time.perf_counter()

    kwargs: dict[str, Any] = {"do_sample": False}
    if lane == "reasoning_aware":
        decoding = lane_cfg["decoding"]
        kwargs = {
            "do_sample": bool(decoding["do_sample"]),
            "temperature": float(decoding["temperature"]),
            "top_p": float(decoding["top_p"]),
        }
        if decoding.get("top_k") is not None:
            kwargs["top_k"] = int(decoding["top_k"])

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            **kwargs,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
            stopping_criteria=StoppingCriteriaList([clock]),
            return_dict_in_generate=True,
            use_cache=True,
        )
    torch.cuda.synchronize()
    finished = time.perf_counter()
    continuation = generated.sequences[0][input_width:]
    raw = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    eos_ids = set(stop_token_ids)
    generated_tokens = int(continuation.shape[0])
    final_token_id = int(continuation[-1]) if generated_tokens else None
    stop_token_id = final_token_id if final_token_id in eos_ids else None
    stop_token_text = (
        tokenizer.decode([stop_token_id], skip_special_tokens=False)
        if stop_token_id is not None
        else None
    )
    if clock.deadline_hit:
        finish_reason = "runtime_deadline"
    elif stop_token_id is not None:
        finish_reason = "eos"
    elif generated_tokens >= int(max_new_tokens):
        finish_reason = "max_tokens"
    else:
        finish_reason = "stopped"
    total = finished - started
    ttft = (clock.first_token_time - started) if clock.first_token_time is not None else total
    return {
        "output": raw,
        "prompt_tokens": int(encoded["attention_mask"].sum().item()),
        "generated_tokens": generated_tokens,
        "max_new_tokens": int(max_new_tokens),
        "finish_reason": finish_reason,
        "configured_eos_token_ids": stop_token_ids,
        "stop_token_id": stop_token_id,
        "stop_token_text": stop_token_text,
        "runtime_deadline_hit": bool(clock.deadline_hit),
        "ttft_seconds": ttft,
        "total_latency_seconds": total,
        "tokens_per_second": generated_tokens / total if total > 0 else 0.0,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "allocated_vram_bytes_after": int(torch.cuda.memory_allocated()),
    }


def semantic_task(semantic: dict[str, Any], task_id: str) -> dict[str, Any]:
    for row in base.semantic_tasks(semantic):
        if str(row.get("task_id")) == task_id:
            return row
    raise RuntimeError(f"semantic task absent: {task_id}")


def topology_case(topology: dict[str, Any], case_id: str) -> dict[str, Any]:
    for row in topology["cases"]:
        if str(row.get("case_id")) == case_id:
            return row
    raise RuntimeError(f"topology case absent: {case_id}")


def score_probe(
    probe: dict[str, Any],
    *,
    output: str,
    semantic: dict[str, Any],
    topology: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], bool | None]:
    if probe["source"] == "semantic":
        task = semantic_task(semantic, str(probe["task_id"]))
        score = evaluate_behavioral_sample(task, output, seed).to_dict()
        hard_checks = [check for check in score["checks"] if check.get("hard")]
        passed = bool(hard_checks) and all(bool(check.get("passed")) for check in hard_checks)
        score["hard_check_count"] = len(hard_checks)
        score["hard_checks_passed"] = passed
        return score, passed

    case = topology_case(topology, str(probe["case_id"]))
    score = topology_runner.score_response(topology, case, output)
    threshold = float(probe.get("minimum_quality_100", 70.0))
    passed = bool(score["schema_compliant"]) and float(score["quality_100"]) >= threshold
    score["screen_minimum_quality_100"] = threshold
    return score, passed


def screen_disposition(rows: list[dict[str, Any]], expected_count: int) -> tuple[str, bool]:
    if len(rows) != expected_count:
        return "screen_inconclusive_incomplete", False
    if any(row["generation"]["runtime_deadline_hit"] for row in rows):
        return "screen_inconclusive_runtime_deadline", False
    if any(row["lane"] == "reasoning_aware" and row["generation"]["finish_reason"] == "max_tokens" for row in rows):
        return "screen_inconclusive_reasoning_budget_exhausted", False
    if all(row.get("probe_pass") is True for row in rows):
        return "screen_passed_eligible_for_tiny_adaptation_only", True
    return "screen_not_promising_under_bounded_probe", False


def main() -> int:
    screen = json.loads(SCREEN_PATH.read_text(encoding="utf-8"))
    recovery = json.loads((ROOT / screen["sources"]["recovery_contract_path"]).read_text(encoding="utf-8"))
    semantic = json.loads((ROOT / screen["sources"]["semantic_pack_path"]).read_text(encoding="utf-8"))
    topology = json.loads((ROOT / screen["sources"]["topology_pack_path"]).read_text(encoding="utf-8"))
    model_id = screen["model"]["model_id"]
    candidate = next((row for row in recovery["candidates"] if row["model_id"] == model_id), None)
    if candidate is None:
        raise RuntimeError("pinned GLM candidate is absent from recovery protocol")
    for key in ("revision", "model_type", "loader"):
        if candidate[key] != screen["model"][key]:
            raise RuntimeError(f"cheap-screen model identity drift: {key}")

    run_id = required("HEPHAESTUS_GLM_SCREEN_RUN_ID")
    repo_sha = required("HEPHAESTUS_REPO_SHA")
    prefix = f"{screen['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    client = base.s3_client()
    seed = int(screen["generation"]["seed"])
    hard_wall_seconds = int(screen["execution"]["hard_wall_seconds"])
    deadline = time.monotonic() + hard_wall_seconds
    model = tokenizer = None
    rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "result_version": "hephaestus-glm-cheap-screen.v1",
        "protocol_id": screen["protocol_id"],
        "status": "running",
        "screening_only": True,
        "certification_claim_allowed": False,
        "training_performed": False,
        "promotion_performed": False,
        "lineage_mutated": False,
        "run_id": run_id,
        "repo_sha": repo_sha,
        "model_id": model_id,
        "revision": candidate["revision"],
        "seed": seed,
        "s3_prefix": prefix,
        "started_at_unix": time.time(),
    }

    def heartbeat(stage: str, **extra: Any) -> None:
        payload = {
            "heartbeat_version": "hephaestus-glm-cheap-screen-heartbeat.v1",
            "run_id": run_id,
            "model_id": model_id,
            "stage": stage,
            "completed_samples": len(rows),
            "expected_samples": len(screen["probes"]),
            "timestamp_unix": time.time(),
            "remaining_internal_wall_seconds": max(0.0, deadline - time.monotonic()),
            **extra,
        }
        sequence = len(rows)
        put_json_verified(client, f"{prefix}/heartbeats/{sequence:03d}-{stage}.json", payload)
        put_json_verified(client, f"{prefix}/progress.json", payload)
        print("GLM_CHEAP_SCREEN_HEARTBEAT_JSON " + json.dumps(payload, sort_keys=True), flush=True)

    try:
        put_json_verified(client, f"{prefix}/run_manifest.json", {
            "protocol_id": screen["protocol_id"],
            "repo_sha": repo_sha,
            "model": screen["model"],
            "generation": screen["generation"],
            "probes": screen["probes"],
            "execution_limits": {
                "hard_wall_seconds": hard_wall_seconds,
                "max_hourly_usd": screen["execution"]["max_hourly_usd"],
                "max_estimated_total_usd": screen["execution"]["max_estimated_total_usd"],
            },
            "training_performed": False,
            "screening_only": True,
        })
        heartbeat("materializing_model")
        snapshot, manifest = base.materialize_model(candidate, Path("/opt/hephaestus-glm-cheap-screen") / run_id / "model")
        put_json_verified(client, f"{prefix}/model_manifest.json", manifest)
        heartbeat("loading_model")
        model, tokenizer, runtime = load_bf16_base(snapshot, candidate, screen)
        put_json_verified(client, f"{prefix}/runtime.json", runtime)
        heartbeat("model_loaded", allocated_memory_bytes=runtime["allocated_memory_bytes_after_load"])

        topology_runner._warmup(model, tokenizer)
        heartbeat("warmup_complete")

        for index, probe in enumerate(screen["probes"], start=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("cheap-screen internal hard wall reached before next probe")
            source = str(probe["source"])
            if source == "semantic":
                prompt = str(semantic_task(semantic, str(probe["task_id"]))["prompt"])
            elif source == "topology":
                case = topology_case(topology, str(probe["case_id"]))
                prompt = topology_runner._prompt(topology, case)
            else:
                raise RuntimeError(f"unsupported cheap-screen probe source: {source}")

            heartbeat("probe_started", probe_id=probe["probe_id"], probe_index=index)
            generated = generate_one(
                model,
                tokenizer,
                prompt,
                candidate=candidate,
                lane=str(probe["lane"]),
                seed=seed,
                max_new_tokens=int(probe["max_new_tokens"]),
                deadline_monotonic=deadline,
                screen=screen,
            )
            projected = project_reasoning(str(generated["output"]), str(probe["lane"]))
            score, probe_pass = score_probe(
                probe,
                output=projected,
                semantic=semantic,
                topology=topology,
                seed=seed,
            )
            if probe["lane"] == "reasoning_aware" and generated["finish_reason"] in {"max_tokens", "runtime_deadline"}:
                probe_pass = None
            row = {
                "sample_version": "hephaestus-glm-cheap-screen-sample.v1",
                "probe_index": index,
                "probe_id": probe["probe_id"],
                "source": source,
                "task_id": probe.get("task_id"),
                "case_id": probe.get("case_id"),
                "role": probe.get("role"),
                "lane": probe["lane"],
                "seed": seed,
                "max_new_tokens": int(probe["max_new_tokens"]),
                "raw_output": generated.pop("output"),
                "output": projected,
                "reasoning_projection_applied": projected != str(generated.get("output", "")).strip(),
                "generation": generated,
                "score": score,
                "probe_pass": probe_pass,
                "screening_only": True,
            }
            # Correct projection flag after raw output has been removed from generation.
            row["reasoning_projection_applied"] = row["output"] != row["raw_output"].strip()
            rows.append(row)
            put_json_verified(client, f"{prefix}/samples/{index:03d}-{probe['probe_id']}.json", row)
            heartbeat(
                "probe_complete",
                probe_id=probe["probe_id"],
                probe_index=index,
                probe_pass=probe_pass,
                finish_reason=generated["finish_reason"],
                generated_tokens=generated["generated_tokens"],
            )
            if generated["runtime_deadline_hit"]:
                break

        disposition, advance = screen_disposition(rows, len(screen["probes"]))
        result.update({
            "status": "completed",
            "disposition": disposition,
            "advance_to_tiny_adaptation": advance,
            "full_lora_recovery_allowed_by_this_result": False,
            "completed_samples": len(rows),
            "expected_samples": len(screen["probes"]),
            "samples": [
                {
                    "probe_id": row["probe_id"],
                    "lane": row["lane"],
                    "probe_pass": row["probe_pass"],
                    "finish_reason": row["generation"]["finish_reason"],
                    "generated_tokens": row["generation"]["generated_tokens"],
                    "total_latency_seconds": row["generation"]["total_latency_seconds"],
                }
                for row in rows
            ],
        })
        print("GLM_CHEAP_SCREEN_RESULT_JSON " + json.dumps({
            "status": result["status"],
            "disposition": disposition,
            "advance_to_tiny_adaptation": advance,
            "completed_samples": len(rows),
            "expected_samples": len(screen["probes"]),
        }, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        result.update({
            "status": "failed",
            "disposition": "screen_runtime_or_evidence_failure",
            "advance_to_tiny_adaptation": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "completed_samples": len(rows),
            "expected_samples": len(screen["probes"]),
        })
        raise
    finally:
        result["completed_at_unix"] = time.time()
        put_json_verified(client, f"{prefix}/result.json", result)
        try:
            del model, tokenizer
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())