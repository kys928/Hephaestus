#!/usr/bin/env python3
"""Bounded BF16 fit probe for Diagnostic Scaling Recovery large-model candidates.

This is infrastructure evidence only. It does not emit a scientific candidate score,
does not mutate the frozen training/evaluation protocol, and never promotes/updates
lineage. It measures whether the exact pinned BF16 base + governed LoRA surface can
fit and execute on a cheaper single GPU.
"""
from __future__ import annotations

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

import run_adaptation_elasticity_v1 as elastic
import run_diagnostic_scaling_v1 as base

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/experiments/hephaestus_diagnostic_scaling_recovery_v1.json"
WORK_ROOT = Path("/opt/hephaestus-cost-fit-probe")


def required(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def main() -> int:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    model_id = required("HEPHAESTUS_MODEL_ID")
    result_key = required("HEPHAESTUS_COST_PROBE_KEY")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    candidate = next((row for row in contract["candidates"] if row["model_id"] == model_id), None)
    if candidate is None:
        raise RuntimeError(f"candidate not in recovery cohort: {model_id}")

    client = base.s3_client()
    result: dict[str, Any] = {
        "probe_version": "diagnostic-scaling-large-model-fit.v1",
        "status": "running",
        "model_id": model_id,
        "revision": candidate["revision"],
        "scientific_result": False,
        "promotion_performed": False,
        "lineage_mutated": False,
        "started_at_unix": time.time(),
    }
    model = tokenizer = peft_model = None
    try:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        snapshot, snapshot_manifest = base.materialize_model(candidate, WORK_ROOT / "model")
        result["snapshot_bytes"] = int(snapshot_manifest["snapshot_total_bytes"])

        # Intentionally bypass only the recovery candidate's conservative routing
        # floor. The shared training loader still requires an >=80GB-class GPU and
        # preserves exact BF16, no quantization, no CPU/disk offload, and one GPU.
        model, tokenizer, runtime = elastic._load_training_base(snapshot, candidate, contract)
        result["runtime"] = runtime
        result["base_allocated_bytes"] = int(torch.cuda.memory_allocated())
        result["base_reserved_bytes"] = int(torch.cuda.memory_reserved())

        target_modules, target_parameters, rank_pattern = base.target_surface(model, candidate, contract)
        train = contract["training"]
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
        model = None
        trainable = [p for p in peft_model.parameters() if p.requires_grad]
        result["trainable_parameters"] = int(sum(p.numel() for p in trainable))
        result["target_module_count"] = len(target_modules)
        result["target_parameter_count"] = len(target_parameters)

        # Worst-case training-shape memory probe: batch=1, seq=1024, optimizer state
        # materialized by an actual step. Token values are synthetic because this is
        # a memory/runtime probe, not scientific training.
        seq = int(train["max_seq_length"])
        vocab = int(getattr(peft_model.config, "vocab_size"))
        input_ids = torch.randint(0, vocab, (1, seq), device="cuda", dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, : seq // 2] = -100
        optimizer = torch.optim.AdamW(trainable, lr=float(train["learning_rate"]), weight_decay=float(train["weight_decay"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peft_model.train()
        started = time.perf_counter()
        out = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        loss = out.loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite loss in fit probe")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        result["training_probe"] = {
            "loss": float(loss.detach().item()),
            "seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        del out, loss, optimizer, input_ids, attention_mask, labels
        torch.cuda.empty_cache()

        # Verify cached generation works on the same fitted LoRA model. This is
        # deliberately bounded; the scientific run retains its frozen token ladders.
        peft_model.eval()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Return a concise JSON object with keys decision, confidence, and rationale."}],
            tokenize=False,
            add_generation_prompt=True,
            **dict(candidate["reasoning_aware"].get("chat_template_kwargs") or {}),
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = peft_model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
            )
        torch.cuda.synchronize()
        generated_tokens = int(generated.sequences.shape[1] - encoded["input_ids"].shape[1])
        result["generation_probe"] = {
            "generated_tokens": generated_tokens,
            "seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "use_cache": True,
        }

        props = torch.cuda.get_device_properties(0)
        total = int(props.total_memory)
        worst_peak = max(
            int(result["training_probe"]["peak_reserved_bytes"]),
            int(result["generation_probe"]["peak_reserved_bytes"]),
        )
        result["gpu_total_bytes"] = total
        result["worst_observed_reserved_bytes"] = worst_peak
        result["observed_headroom_bytes"] = total - worst_peak
        result["observed_headroom_gib"] = (total - worst_peak) / (1024 ** 3)
        result["status"] = "fit_probe_passed"
        print("HEPHAESTUS_LARGE_MODEL_FIT_PROBE_JSON " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        result["status"] = "fit_probe_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print("HEPHAESTUS_LARGE_MODEL_FIT_PROBE_JSON " + json.dumps({k:v for k,v in result.items() if k != "traceback"}, sort_keys=True), flush=True)
        raise
    finally:
        result["completed_at_unix"] = time.time()
        raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        client.put_object(Bucket=base.bucket(), Key=result_key, Body=raw)
        observed = base.read_s3(client, result_key)
        if observed != raw:
            raise RuntimeError("large-model fit probe S3 readback mismatch")
        try:
            del peft_model, model, tokenizer
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
