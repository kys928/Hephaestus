#!/usr/bin/env python3
"""Compute-efficient execution path for the frozen Interface Repair V1 science.

V2 changes execution mechanics only: dynamic-length microbatches, contract-aware
JSON early stopping, sharded S3 evidence, immutable baseline reuse, stage timing,
and a bounded planner preflight benchmark.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_interface_repair_v1 as v1

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_repair_v2.json"
REQUIRED_KEYS = {"decision", "action", "primary_variable", "confidence", "evidence_refs", "uncertainties", "rationale"}


def load_cfg() -> dict[str, Any]:
    value = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("V2 config must be an object")
    return value


def put_bytes_verified(client: Any, key: str, raw: bytes, content_type: str) -> dict[str, Any]:
    digest = hashlib.sha256(raw).hexdigest()
    client.put_object(
        Bucket=v1.bucket(), Key=key, Body=raw, ContentType=content_type,
        Metadata={"sha256": digest},
    )
    head = client.head_object(Bucket=v1.bucket(), Key=key)
    observed_digest = str((head.get("Metadata") or {}).get("sha256", ""))
    if int(head["ContentLength"]) != len(raw) or observed_digest != digest:
        raise RuntimeError(f"S3 metadata verification failed for {key}")
    return {"s3_key": key, "sha256": digest, "bytes": len(raw)}


def put_json(client: Any, key: str, payload: object) -> dict[str, Any]:
    raw = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return put_bytes_verified(client, key, raw, "application/json")


def put_jsonl(client: Any, key: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = ("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")
    return put_bytes_verified(client, key, raw, "application/x-ndjson")


def get_json_optional(client: Any, key: str) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=v1.bucket(), Key=key)
    except Exception as exc:  # noqa: BLE001
        data = getattr(exc, "response", {})
        code = str(data.get("Error", {}).get("Code", "")) if isinstance(data, dict) else ""
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    try:
        value = json.loads(response["Body"].read().decode("utf-8"))
    finally:
        response["Body"].close()
    if not isinstance(value, dict):
        raise RuntimeError(f"{key} is not a JSON object")
    return value


def contract_json(raw: str, delimiters: Sequence[str]) -> dict[str, Any] | None:
    projected = v1.project_reasoning(raw, delimiters)
    obj, _ = v1.extract_json(projected)
    return obj if isinstance(obj, dict) and set(obj) == REQUIRED_KEYS else None


def generate(model: Any, tokenizer: Any, runtime: Mapping[str, Any], text: str, seed: int, deadline: float) -> dict[str, Any]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    encoded = v1.encode(tokenizer, str(runtime["load_kind"]), text)
    width = int(encoded["input_ids"].shape[-1])
    delimiters = [str(x) for x in runtime.get("reasoning_projection", [])]

    class Deadline(StoppingCriteria):
        def __init__(self) -> None:
            self.hit = False
        def __call__(self, input_ids, scores, **kwargs):  # type: ignore[no-untyped-def]
            del input_ids, scores, kwargs
            if time.monotonic() >= deadline:
                self.hit = True
                return True
            return False

    class ContractComplete(StoppingCriteria):
        def __init__(self) -> None:
            self.hit = False
            self._last_checked = 0
        def __call__(self, input_ids, scores, **kwargs):  # type: ignore[no-untyped-def]
            del scores, kwargs
            continuation = input_ids[0][width:]
            n = int(continuation.shape[0])
            if n < 8 or (n - self._last_checked) < 4:
                return False
            self._last_checked = n
            raw = tokenizer.decode(continuation, skip_special_tokens=True)
            if contract_json(str(raw), delimiters) is not None:
                self.hit = True
                return True
            return False

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    deadline_stop = Deadline()
    contract_stop = ContractComplete()
    started = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            **dict(runtime["decoding"]),
            max_new_tokens=int(runtime["max_new_tokens"]),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=v1.stop_ids(model, tokenizer),
            stopping_criteria=StoppingCriteriaList([deadline_stop, contract_stop]),
            return_dict_in_generate=True,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    continuation = out.sequences[0][width:]
    raw = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    projected = v1.project_reasoning(raw, delimiters)
    return {
        "raw_output": raw,
        "projected_output": projected,
        "prompt_tokens": width,
        "generated_tokens": int(continuation.shape[0]),
        "latency_seconds": elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "deadline_hit": deadline_stop.hit,
        "contract_early_stop": contract_stop.hit,
        "complete_contract_json": contract_json(raw, delimiters) is not None,
    }


def training_example(tokenizer: Any, runtime: Mapping[str, Any], pack: Mapping[str, Any], case: Mapping[str, Any], role: str, maxlen: int) -> dict[str, Any]:
    import torch

    q = v1.prompt(pack, case, role)
    a = v1.target_answer(case)
    p = v1.template_ids(tokenizer, str(runtime["load_kind"]), [{"role": "user", "content": q}], True)
    f = v1.template_ids(tokenizer, str(runtime["load_kind"]), [{"role": "user", "content": q}, {"role": "assistant", "content": a}], False)
    lcp = 0
    for x, y in zip(p, f):
        if x != y:
            break
        lcp += 1
    if lcp < 8:
        raise RuntimeError(f"chat-template prefix mismatch {role} {case['case_id']} lcp={lcp}")
    if len(f) > maxlen:
        raise RuntimeError(f"training example exceeds max length {role} {case['case_id']} {len(f)}>{maxlen}")
    labels = [-100] * min(lcp, len(f)) + f[min(lcp, len(f)):]
    return {
        "input_ids": torch.tensor([f]),
        "attention_mask": torch.ones((1, len(f)), dtype=torch.long),
        "labels": torch.tensor([labels]),
        "nonpad_tokens": len(f),
        "supervised_tokens": sum(x != -100 for x in labels),
        "token_slots": len(f),
    }


def train(
    model: Any,
    tokenizer: Any,
    runtime: Mapping[str, Any],
    pack: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    role: str,
    cfg: Mapping[str, Any],
    deadline: float,
    heartbeat,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    import torch

    tr = cfg["training"]
    maxlen = int(tr["max_sequence_length"])
    steps = int(steps_override if steps_override is not None else tr["optimizer_steps"][role])
    accum = int(tr["gradient_accumulation_steps"])
    required_cases = steps * accum
    if len(cases) < required_cases:
        raise RuntimeError(f"insufficient training cases: {len(cases)} < {required_cases}")
    chosen = list(cases[:required_cases]) if steps_override is not None else list(cases)
    if len(chosen) != required_cases:
        raise RuntimeError(f"training geometry mismatch: {steps}*{accum}!={len(chosen)}")
    examples = [training_example(tokenizer, runtime, pack, c, role, maxlen) for c in chosen]
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if tr["gradient_checkpointing"]:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"]))
    order = list(range(len(examples)))
    random.Random(int(tr["seed"])).shuffle(order)
    cursor = 0
    losses: list[float] = []
    supervised = nonpad = slots = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError("interface repair V2 hard wall during training")
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            ex = examples[order[cursor]]
            cursor += 1
            batch = {k: v.to("cuda", non_blocking=True) for k, v in ex.items() if k in {"input_ids", "attention_mask", "labels"}}
            output = model(**batch, use_cache=False)
            loss = output.loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at optimizer step {step}")
            loss.backward()
            total += float(loss.detach().cpu()) * accum
            supervised += int(ex["supervised_tokens"])
            nonpad += int(ex["nonpad_tokens"])
            slots += int(ex["token_slots"])
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], float(tr["max_grad_norm"]))
        optimizer.step()
        losses.append(total / accum)
        if step == 1 or step % 20 == 0 or step == steps:
            heartbeat("training_progress", optimizer_step=step, optimizer_steps=steps, loss=losses[-1], token_slots=slots)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()
    return {
        "optimizer_steps": steps,
        "gradient_accumulation_steps": accum,
        "dynamic_token_slots": slots,
        "supervised_token_updates": supervised,
        "nonpad_token_updates": nonpad,
        "training_seconds": seconds,
        "seconds_per_optimizer_step": seconds / max(1, steps),
        "peak_training_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
    }


def evaluate(
    model: Any,
    tokenizer: Any,
    runtime: Mapping[str, Any],
    pack: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    role: str,
    seed: int,
    deadline: float,
    client: Any,
    prefix: str,
    phase: str,
    heartbeat,
    shard_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    shard: list[dict[str, Any]] = []
    shard_index = 0
    for i, case in enumerate(cases, 1):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"interface repair V2 hard wall during {phase}")
        gen = generate(model, tokenizer, runtime, v1.prompt(pack, case, role), seed + i, deadline)
        score = v1.score_case(pack, case, role, str(gen["projected_output"]))
        row = {
            "case_id": case["case_id"], "role": role, "phase": phase, "kind": case.get("kind"),
            "generation": gen, "score": score, "expected": case["expected"],
        }
        rows.append(row)
        shard.append(row)
        if len(shard) >= shard_size or i == len(cases):
            shard_index += 1
            put_jsonl(client, f"{prefix}/samples/{phase}/shard-{shard_index:04d}.jsonl", shard)
            shard = []
        if i == 1 or i % 32 == 0 or i == len(cases):
            heartbeat(
                f"{phase}_progress", completed=i, total=len(cases),
                quality_100=v1.summarize(rows)["quality_100"],
                generated_tokens=sum(int(r["generation"]["generated_tokens"]) for r in rows),
            )
    return rows, v1.summarize(rows)


def baseline_identity(cfg: Mapping[str, Any], role: str, role_spec: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    identity = {
        "pack_sha256": cfg["pack"]["canonical_sha256"],
        "role": role,
        "model_id": role_spec["model_id"],
        "revision": role_spec["revision"],
        "parent_adapter_sha256": role_spec["adapter"]["sha256"],
        "seed": cfg["training"]["seed"],
        "runtime": cfg["roles"][role],
        "certification_cases": cfg["pack"]["certification_cases_per_role"],
        "regression_cases": cfg["pack"]["regression_cases_per_role"],
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), identity


def load_or_compute_baseline(
    *, client: Any, cfg: Mapping[str, Any], pack: Mapping[str, Any], role: str,
    role_spec: Mapping[str, Any], model: Any, tokenizer: Any, runtime: Mapping[str, Any],
    deadline: float, heartbeat, shard_size: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    digest, identity = baseline_identity(cfg, role, role_spec)
    cache_prefix = f"{str(cfg['execution']['baseline_cache_prefix']).rstrip('/')}/{role}/{digest}"
    summary_key = f"{cache_prefix}/summary.json"
    cached = get_json_optional(client, summary_key) if cfg["execution"].get("baseline_cache_enabled") else None
    if cached is not None:
        if cached.get("identity") != identity:
            raise RuntimeError("baseline cache identity mismatch")
        heartbeat("baseline_cache_hit", baseline_cache_key=summary_key)
        return dict(cached["pre_interface"]), dict(cached["pre_regression"]), {"hit": True, "key": summary_key, "digest": digest}

    heartbeat("pre_interface_certification_started", baseline_cache_hit=False)
    _, pre_interface = evaluate(
        model, tokenizer, runtime, pack, pack["partitions"][role]["certification"], role,
        int(cfg["training"]["seed"]), deadline, client, cache_prefix, "pre_interface", heartbeat, shard_size,
    )
    heartbeat("pre_regression_started", baseline_cache_hit=False)
    _, pre_regression = evaluate(
        model, tokenizer, runtime, pack, pack["partitions"][role]["regression"], role,
        int(cfg["training"]["seed"]) + 10000, deadline, client, cache_prefix, "pre_regression", heartbeat, shard_size,
    )
    payload = {"identity": identity, "pre_interface": pre_interface, "pre_regression": pre_regression, "created_at_unix": time.time()}
    put_json(client, summary_key, payload)
    return pre_interface, pre_regression, {"hit": False, "key": summary_key, "digest": digest}


def run_preflight(
    *, cfg: Mapping[str, Any], pack: Mapping[str, Any], role: str, model: Any, tokenizer: Any,
    runtime: Mapping[str, Any], deadline: float, client: Any, prefix: str, heartbeat,
) -> dict[str, Any]:
    n_eval = int(cfg["preflight"]["evaluation_cases"])
    n_steps = int(cfg["preflight"]["optimizer_steps"])
    cert = list(pack["partitions"][role]["certification"])
    if n_eval > len(cert):
        raise RuntimeError("preflight evaluation sample exceeds certification partition")
    indexes = [min(len(cert) - 1, (i * len(cert)) // n_eval) for i in range(n_eval)]
    sample = [cert[i] for i in indexes]
    heartbeat("preflight_evaluation_started", sample_count=n_eval)
    rows, _ = evaluate(
        model, tokenizer, runtime, pack, sample, role, int(cfg["training"]["seed"]), deadline,
        client, prefix, "preflight_eval", heartbeat, int(cfg["execution"]["evaluation_shard_size"]),
    )
    eval_seconds = sum(float(r["generation"]["latency_seconds"]) for r in rows)
    complete_rate = sum(bool(r["generation"]["complete_contract_json"]) for r in rows) / max(1, len(rows))
    deadline_hits = sum(bool(r["generation"]["deadline_hit"]) for r in rows)
    heartbeat("preflight_training_started", optimizer_steps=n_steps)
    training = train(
        model, tokenizer, runtime, pack, pack["partitions"][role]["train"], role, cfg, deadline, heartbeat,
        steps_override=n_steps,
    )
    mean_eval_seconds = eval_seconds / max(1, len(rows))
    projected_eval_seconds = mean_eval_seconds * (2 * int(cfg["pack"]["certification_cases_per_role"]) + 2 * int(cfg["pack"]["regression_cases_per_role"]))
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
        "sampled_evaluation_cases": n_eval,
        "sampled_optimizer_steps": n_steps,
        "mean_evaluation_seconds_per_case": mean_eval_seconds,
        "mean_generated_tokens": sum(int(r["generation"]["generated_tokens"]) for r in rows) / max(1, len(rows)),
        "contract_early_stop_rate": sum(bool(r["generation"]["contract_early_stop"]) for r in rows) / max(1, len(rows)),
        "complete_contract_json_rate": complete_rate,
        "deadline_hits": deadline_hits,
        "training": training,
        "projected_full_role_evaluation_seconds": projected_eval_seconds,
        "projected_full_role_training_seconds": projected_train_seconds,
        "projected_full_role_seconds": projected_role_seconds,
        "maximum_projected_role_seconds": cfg["preflight"]["maximum_projected_role_seconds"],
        "approved_for_full_run": approved,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg()
    pack = v1.load_pack(cfg)
    stack = v1.load_stack(cfg)
    role = v1.required("HEPHAESTUS_REPAIR_ROLE")
    if role not in cfg["trainable_roles"] or role == "diagnosis":
        raise RuntimeError(f"role {role} is not trainable in Interface Repair V2")
    if args.preflight and role != str(cfg["preflight"]["role"]):
        raise RuntimeError("V2 preflight is pinned to the configured bottleneck role")

    run_id = v1.required("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID")
    repo_sha = v1.required("HEPHAESTUS_REPO_SHA")
    client = v1.s3_client()
    prefix = f"{str(cfg['execution']['s3_prefix']).rstrip('/')}/{run_id}/{'preflight' if args.preflight else 'roles'}/{role}"
    wall = int(cfg["execution"]["hard_wall_seconds_preflight"] if args.preflight else cfg["execution"]["hard_wall_seconds_per_role"])
    deadline = time.monotonic() + wall
    role_spec = stack["roles"][role]
    root = Path("/opt/hephaestus-interface-repair-v2") / run_id / role
    telemetry: dict[str, float] = {}
    result: dict[str, Any] = {
        "result_version": "hephaestus-interface-repair-execution.v2",
        "status": "running", "role": role, "run_id": run_id, "repo_sha": repo_sha,
        "pack_sha256": cfg["pack"]["canonical_sha256"], "source_role_mastery_run_id": stack["source_run_id"],
        "parent_adapter_sha256": role_spec["adapter"]["sha256"], "preflight": args.preflight,
        "diagnosis_mutated": False, "production_promotion_performed": False,
    }

    def heartbeat(stage: str, **extra: object) -> None:
        payload = {
            "stage": stage, "role": role, "run_id": run_id, "timestamp_unix": time.time(),
            "remaining_wall_seconds": max(0.0, deadline - time.monotonic()), **extra,
        }
        put_json(client, f"{prefix}/progress.json", payload)
        print("INTERFACE_REPAIR_V2_HEARTBEAT_JSON " + json.dumps(payload, sort_keys=True), flush=True)

    try:
        put_json(client, f"{prefix}/run_manifest.json", {"protocol": cfg, "role": role, "repo_sha": repo_sha, "parent_role_spec": role_spec})
        started = time.perf_counter()
        heartbeat("materializing_parent")
        base_dir, adapter_dir = v1.materialize_parent(client, role_spec, root / "model")
        telemetry["materialize_parent_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        model, tokenizer, runtime_info = v1.load_model(role, cfg, role_spec, base_dir, adapter_dir)
        telemetry["model_load_seconds"] = time.perf_counter() - started
        put_json(client, f"{prefix}/runtime.json", runtime_info)
        runtime = cfg["roles"][role]

        if args.preflight:
            started = time.perf_counter()
            preflight = run_preflight(
                cfg=cfg, pack=pack, role=role, model=model, tokenizer=tokenizer, runtime=runtime,
                deadline=deadline, client=client, prefix=prefix, heartbeat=heartbeat,
            )
            telemetry["preflight_body_seconds"] = time.perf_counter() - started
            result.update(preflight)
            result["runtime"] = runtime_info
            result["telemetry"] = telemetry
            result["completed_at_unix"] = time.time()
            put_json(client, f"{prefix}/result.json", result)
            heartbeat("preflight_complete", approved_for_full_run=preflight["approved_for_full_run"], projected_full_role_seconds=preflight["projected_full_role_seconds"])
            print("INTERFACE_REPAIR_V2_PREFLIGHT_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
            return 0

        shard_size = int(cfg["execution"]["evaluation_shard_size"])
        started = time.perf_counter()
        pre_interface, pre_regression, baseline = load_or_compute_baseline(
            client=client, cfg=cfg, pack=pack, role=role, role_spec=role_spec, model=model,
            tokenizer=tokenizer, runtime=runtime, deadline=deadline, heartbeat=heartbeat, shard_size=shard_size,
        )
        telemetry["baseline_seconds"] = time.perf_counter() - started

        heartbeat("training_started", pre_interface_quality_100=pre_interface["quality_100"], pre_regression_quality_100=pre_regression["quality_100"], baseline_cache_hit=baseline["hit"])
        started = time.perf_counter()
        training = train(model, tokenizer, runtime, pack, pack["partitions"][role]["train"], role, cfg, deadline, heartbeat)
        telemetry["training_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        artifact = v1.save_adapter(model, root, client, f"{prefix}/selected-adapter.tar.gz")
        telemetry["adapter_persist_seconds"] = time.perf_counter() - started

        heartbeat("post_interface_certification_started")
        started = time.perf_counter()
        _, post_interface = evaluate(
            model, tokenizer, runtime, pack, pack["partitions"][role]["certification"], role,
            int(cfg["training"]["seed"]), deadline, client, prefix, "post_interface", heartbeat, shard_size,
        )
        telemetry["post_interface_seconds"] = time.perf_counter() - started

        heartbeat("post_regression_started")
        started = time.perf_counter()
        _, post_regression = evaluate(
            model, tokenizer, runtime, pack, pack["partitions"][role]["regression"], role,
            int(cfg["training"]["seed"]) + 10000, deadline, client, prefix, "post_regression", heartbeat, shard_size,
        )
        telemetry["post_regression_seconds"] = time.perf_counter() - started

        certification = v1.certify(pre_interface, post_interface, pre_regression, post_regression, cfg)
        result.update({
            "status": "completed", "runtime": runtime_info,
            "pre_interface": pre_interface, "post_interface": post_interface,
            "pre_regression": pre_regression, "post_regression": post_regression,
            "baseline_cache": baseline, "training": training, "telemetry": telemetry,
            "interface_quality_gain_100": post_interface["quality_100"] - pre_interface["quality_100"],
            "certification": certification, "adapter": artifact, "completed_at_unix": time.time(),
        })
        put_json(client, f"{prefix}/result.json", result)
        heartbeat("complete", certified=certification["certified"], post_interface_quality_100=post_interface["quality_100"], post_regression_quality_100=post_regression["quality_100"], adapter_sha256=artifact["sha256"])
        print("INTERFACE_REPAIR_V2_ROLE_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        result.update({
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "telemetry": telemetry, "completed_at_unix": time.time(),
        })
        try:
            put_json(client, f"{prefix}/result.json", result)
        except Exception:
            pass
        print("INTERFACE_REPAIR_V2_ROLE_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        raise
    finally:
        shutil.rmtree(root, ignore_errors=True)
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
