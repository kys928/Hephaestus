#!/usr/bin/env python3
"""Train identical low-dose role LoRAs and measure adaptation elasticity."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cognitive_topology_v1 as topology_runner
from build_adaptation_elasticity_dataset_v1 import training_user_prompt

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs/experiments/hephaestus_adaptation_elasticity_v1.json"
TOPOLOGY_PATH = Path(__file__).resolve().parents[1] / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"
SCIENTIFIC_ROOT = Path("/workspace/hephaestus/scientific/v1")


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "--" for ch in value).strip("-.")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _hash_files(root: Path) -> tuple[dict[str, str], int]:
    hashes: dict[str, str] = {}
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        hashes[path.relative_to(root).as_posix()] = "sha256:" + _sha(raw)
        total += len(raw)
    return hashes, total


def _load_protocols() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    protocol_raw = PROTOCOL_PATH.read_bytes()
    topology_raw = TOPOLOGY_PATH.read_bytes()
    return json.loads(protocol_raw), _sha(protocol_raw), json.loads(topology_raw), _sha(topology_raw)


def _paths() -> tuple[Path, Path]:
    run_id = _required("HEPHAESTUS_PROOF_RUN_ID")
    attempt = _required("HEPHAESTUS_ATTEMPT")
    return SCIENTIFIC_ROOT / "adaptation_elasticity" / run_id, SCIENTIFIC_ROOT / "executions" / run_id / f"attempt-{attempt}"


def _load_admission(protocol_sha: str, topology_sha: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repo_sha = _required("HEPHAESTUS_REPO_SHA")
    root = SCIENTIFIC_ROOT / "adaptation_elasticity" / "model_admission" / repo_sha
    admission = json.loads((root / "admission.json").read_text(encoding="utf-8"))
    requirements = (root / "requirements.txt").read_bytes()
    if admission.get("repo_sha") != repo_sha or admission.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("adaptation admission repository/protocol mismatch")
    if admission.get("topology_protocol_sha256") != topology_sha:
        raise RuntimeError("adaptation admission topology hash mismatch")
    if _sha(requirements) != admission.get("requirements_sha256"):
        raise RuntimeError("adaptation dependency lock hash mismatch")
    if admission.get("training_approved") is not True or admission.get("promotion_allowed") is not False or admission.get("lineage_mutation_allowed") is not False:
        raise RuntimeError("adaptation governance admission is invalid")
    contamination = json.loads((root / "contamination_report.json").read_text(encoding="utf-8"))
    if contamination.get("passed") is not True:
        raise RuntimeError("adaptation dataset failed contamination gate")
    dataset_raw = (root / "training.jsonl").read_bytes()
    if _sha(dataset_raw) != admission["dataset_manifest"]["dataset_sha256"]:
        raise RuntimeError("adaptation dataset hash mismatch")
    return admission, [json.loads(line) for line in dataset_raw.decode().splitlines() if line.strip()]


def _load_baseline(protocol: dict[str, Any], topology_sha: str) -> dict[str, Any]:
    path = SCIENTIFIC_ROOT / "cognitive_topology" / protocol["baseline"]["run_id"] / "cohort_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("protocol_sha256") != topology_sha:
        raise RuntimeError("frozen topology baseline is unavailable or mismatched")
    return result


def _load_training_base(snapshot: Path, candidate: dict[str, Any], protocol: dict[str, Any]):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("adaptation elasticity requires exactly one visible CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    if props.total_memory / (1024 ** 3) < 75:
        raise RuntimeError("adaptation elasticity requires an >=80GB-class GPU for comparable 14B LoRA training")
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if config.model_type != candidate["model_type"]:
        raise RuntimeError("training architecture differs from admission")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    loaders = {"AutoModelForCausalLM": AutoModelForCausalLM, "AutoModelForImageTextToText": AutoModelForImageTextToText}
    model = loaders[candidate["loader"]].from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True)
    devices = {str(t.device) for t in (*model.parameters(), *model.buffers())}
    if not devices or not devices <= {"cuda", "cuda:0"}:
        raise RuntimeError(f"forbidden training tensor residency: {sorted(devices)}")
    float_dtypes = {str(p.dtype) for p in model.parameters() if p.is_floating_point()}
    if float_dtypes != {"torch.bfloat16"}:
        raise RuntimeError(f"training base dtype is not uniformly BF16: {sorted(float_dtypes)}")
    if getattr(config, "quantization_config", None):
        raise RuntimeError("quantized base is forbidden in elasticity V1")
    if bool(protocol["training"]["gradient_checkpointing"]):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, {"gpu": props.name, "gpu_memory_bytes": props.total_memory, "torch": torch.__version__, "cuda": torch.version.cuda, "tensor_devices": sorted(devices), "weight_dtypes": sorted(float_dtypes), "loader": candidate["loader"], "cpu_disk_offload": False, "quantization": False, "gradient_checkpointing": bool(protocol["training"]["gradient_checkpointing"]), "allocated_memory_bytes": int(torch.cuda.memory_allocated())}


def _target_modules(model: Any, suffixes: list[str]) -> list[str]:
    import torch
    blocked = ("vision", "visual", "image", "pixel", "projector", "multi_modal", "multimodal")
    targets: list[str] = []
    for name, module in model.named_modules():
        lowered = name.casefold()
        if any(token in lowered for token in blocked):
            continue
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in suffixes:
            targets.append(name)
    if not targets:
        raise RuntimeError("no governed LoRA target modules were found")
    return sorted(targets)


def _tokenize_example(tokenizer: Any, row: dict[str, Any], max_len: int) -> dict[str, list[int]]:
    user = training_user_prompt(row)
    target = json.dumps(row["target"], separators=(",", ":"), ensure_ascii=False)
    user_messages = [{"role": "user", "content": user}]
    full_messages = [*user_messages, {"role": "assistant", "content": target}]
    prompt_text = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    common = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        common += 1
    if common < max(1, len(prompt_ids) - 4):
        raise RuntimeError("chat template prompt/target boundary is not stable enough for assistant-only masking")
    if len(full_ids) > max_len:
        raise RuntimeError(f"training record exceeds max_seq_length: {len(full_ids)}>{max_len}")
    labels = [-100] * common + full_ids[common:]
    if not any(label != -100 for label in labels):
        raise RuntimeError("training record contains no assistant target tokens")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def _tensorize(item: dict[str, list[int]]):
    import torch
    return {key: torch.tensor([value], dtype=torch.long, device="cuda") for key, value in item.items()}


def _adapter_manifest(path: Path, *, model_id: str, role: str, dose: int, trainable_parameters: int, target_modules: list[str]) -> dict[str, Any]:
    hashes, byte_size = _hash_files(path)
    return {"manifest_version": "adaptation-elasticity-lora.v1", "model_id": model_id, "role": role, "dose_epoch": dose, "trainable_parameters": trainable_parameters, "target_module_count": len(target_modules), "target_modules_sha256": _sha("\n".join(target_modules).encode()), "components": hashes, "adapter_bytes": byte_size, "manifest_sha256": _sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode())}


def _evaluate_role(model: Any, tokenizer: Any, topology: dict[str, Any], candidate: dict[str, Any], role: str, runtime: dict[str, Any], sample_path: Path) -> dict[str, Any]:
    role_cases = [case for case in topology["cases"] if case["role"] == role]
    role_spec = dict(topology)
    role_spec["roles"] = [role]
    samples: list[dict[str, Any]] = []
    tokenizer.padding_side = "left"
    model.eval()
    for seed in topology["generation"]["seeds"]:
        for case in role_cases:
            prompt = topology_runner._prompt(topology, case)
            generated = topology_runner._generate_one(model, tokenizer, prompt, seed=int(seed), max_new_tokens=int(topology["generation"]["max_new_tokens"]))
            score = topology_runner.score_response(topology, case, generated["output"])
            row = {"model_id": candidate["model_id"], "revision": candidate["revision"], "case_id": case["case_id"], "role": role, "condition": case["condition"], "pair_id": case.get("pair_id"), "seed": int(seed), "output": generated["output"], "generation": {key: value for key, value in generated.items() if key != "output"}, "score": score}
            samples.append(row)
            _append_jsonl(sample_path, row)
    expected = len(role_cases) * len(topology["generation"]["seeds"])
    if len(samples) != expected:
        raise RuntimeError(f"incomplete adapted role evidence: {len(samples)} != {expected}")
    summary = topology_runner._summarize_model(role_spec, candidate, samples, runtime)
    tokenizer.padding_side = "right"
    model.train()
    return summary["roles"][role]


def _elasticity_record(*, baseline: dict[str, Any], adapted: dict[str, Any], dose: int, training_seconds: float, training_tokens: int, trainable_parameters: int, adapter_manifest: dict[str, Any], training_peak_vram_bytes: int, mean_loss: float) -> dict[str, Any]:
    delta = float(adapted["quality_100"]) - float(baseline["quality_100"])
    gpu_hours = training_seconds / 3600.0
    params_m = trainable_parameters / 1_000_000.0
    tokens_m = training_tokens / 1_000_000.0
    return {"dose_epoch": dose, "baseline_quality_100": baseline["quality_100"], "adapted_quality_100": adapted["quality_100"], "delta_quality_points": delta, "delta_per_gpu_hour": delta / gpu_hours if gpu_hours > 0 else None, "delta_per_million_trainable_parameters": delta / params_m if params_m > 0 else None, "delta_per_million_training_tokens": delta / tokens_m if tokens_m > 0 else None, "schema_compliance_delta": adapted["schema_compliance"] - baseline["schema_compliance"], "evidence_grounding_delta": adapted["evidence_grounding"] - baseline["evidence_grounding"], "confidence_calibration_delta": adapted["confidence_calibration"] - baseline["confidence_calibration"], "hallucination_rate_delta": adapted["hallucination_rate"] - baseline["hallucination_rate"], "training_seconds_cumulative": training_seconds, "gpu_hours_cumulative": gpu_hours, "training_tokens_cumulative": training_tokens, "trainable_parameters": trainable_parameters, "adapter_bytes": adapter_manifest["adapter_bytes"], "training_peak_vram_bytes": training_peak_vram_bytes, "mean_training_loss_epoch": mean_loss, "adapted_role_summary": adapted, "adapter_manifest": adapter_manifest}


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> int:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import get_linear_schedule_with_warmup

    protocol, protocol_sha, topology, topology_sha = _load_protocols()
    if protocol["baseline"]["protocol_sha256"] != topology_sha:
        raise RuntimeError("frozen topology bytes changed after adaptation protocol was admitted")
    admission, dataset = _load_admission(protocol_sha, topology_sha)
    baseline = _load_baseline(protocol, topology_sha)
    proof_root, execution_root = _paths()
    proof_root.mkdir(parents=True, exist_ok=True)
    execution_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(proof_root / "protocol_snapshot.json", protocol)
    _atomic_json(proof_root / "topology_protocol_snapshot.json", topology)

    baseline_map = {row["model_id"]: row for row in baseline["model_summaries"]}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    train = protocol["training"]
    per_role = {role: [row for row in dataset if row["role"] == role] for role in protocol["roles"]}
    for role, rows in per_role.items():
        if len(rows) != int(protocol["dataset"]["examples_per_role"]):
            raise RuntimeError(f"training dataset count drifted for {role}")

    _atomic_json(proof_root / "run_manifest.json", {"run_id": _required("HEPHAESTUS_PROOF_RUN_ID"), "repo_sha": _required("HEPHAESTUS_REPO_SHA"), "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha, "topology_protocol_sha256": topology_sha, "baseline_run_id": protocol["baseline"]["run_id"], "candidate_order": [row["model_id"] for row in topology["candidates"]], "roles": protocol["roles"], "training_method": "lora", "promotion_allowed": False, "lineage_mutation_allowed": False, "started_at_unix": time.time()})

    for candidate in topology["candidates"]:
        base_model = tokenizer = None
        model_slug = _slug(candidate["model_id"])
        try:
            snapshot, model_manifest = topology_runner._materialize(candidate, admission, proof_root)
            base_model, tokenizer, runtime = _load_training_base(snapshot, candidate, protocol)
            runtime.update({"model_id": candidate["model_id"], "revision": candidate["revision"], "manifest_hash": model_manifest["manifest_hash"]})
            target_modules = _target_modules(base_model, list(train["target_module_suffixes"]))
            runtime["lora_target_module_count"] = len(target_modules)
            runtime["lora_target_modules_sha256"] = _sha("\n".join(target_modules).encode())
            _atomic_json(proof_root / "runtime" / f"{model_slug}.json", runtime)
            model_result = {"model_id": candidate["model_id"], "revision": candidate["revision"], "license": candidate["license"], "roles": {}, "runtime": runtime, "status": "complete"}

            for role_index, role in enumerate(protocol["roles"]):
                torch.manual_seed(int(train["model_seed"]) + role_index)
                torch.cuda.manual_seed_all(int(train["model_seed"]) + role_index)
                rows = list(per_role[role])
                random.Random(int(train["shuffle_seed"]) + role_index).shuffle(rows)
                encoded = [_tokenize_example(tokenizer, row, int(train["max_seq_length"])) for row in rows]
                lora_config = LoraConfig(r=int(train["rank"]), lora_alpha=int(train["alpha"]), lora_dropout=float(train["dropout"]), bias=str(train["bias"]), task_type=TaskType.CAUSAL_LM, target_modules=target_modules)
                peft_model = get_peft_model(base_model, lora_config)
                trainable_parameters = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
                if trainable_parameters <= 0:
                    raise RuntimeError(f"no trainable LoRA parameters for {candidate['model_id']} role={role}")
                optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=float(train["learning_rate"]), weight_decay=float(train["weight_decay"]))
                accumulation = int(train["gradient_accumulation_steps"])
                updates_per_epoch = math.ceil(len(encoded) / accumulation)
                total_updates = updates_per_epoch * int(train["epochs"])
                warmup_steps = max(1, int(round(total_updates * float(train["warmup_ratio"]))))
                scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
                cumulative_seconds = 0.0
                cumulative_tokens = 0
                role_doses: list[dict[str, Any]] = []
                optimizer.zero_grad(set_to_none=True)

                for epoch in range(1, int(train["epochs"]) + 1):
                    peft_model.train()
                    tokenizer.padding_side = "right"
                    epoch_losses: list[float] = []
                    epoch_started = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    for idx, item in enumerate(encoded):
                        batch = _tensorize(item)
                        cumulative_tokens += sum(1 for label in item["labels"] if label != -100)
                        outputs = peft_model(**batch)
                        loss = outputs.loss / accumulation
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError(f"non-finite LoRA loss for {candidate['model_id']} role={role}")
                        loss.backward()
                        epoch_losses.append(float(loss.detach().item()) * accumulation)
                        if ((idx + 1) % accumulation == 0) or (idx + 1 == len(encoded)):
                            torch.nn.utils.clip_grad_norm_([p for p in peft_model.parameters() if p.requires_grad], float(train["max_grad_norm"]))
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    cumulative_seconds += time.perf_counter() - epoch_started
                    training_peak = int(torch.cuda.max_memory_allocated())
                    adapter_dir = proof_root / "adapters" / model_slug / role / f"dose-epoch-{epoch}"
                    peft_model.save_pretrained(adapter_dir, safe_serialization=True)
                    adapter_manifest = _adapter_manifest(adapter_dir, model_id=candidate["model_id"], role=role, dose=epoch, trainable_parameters=trainable_parameters, target_modules=target_modules)
                    _atomic_json(adapter_dir / "hephaestus_adapter_manifest.json", adapter_manifest)
                    sample_path = proof_root / "post_samples" / model_slug / role / f"dose-epoch-{epoch}.jsonl"
                    if sample_path.exists():
                        sample_path.unlink()
                    adapted_summary = _evaluate_role(peft_model, tokenizer, topology, candidate, role, runtime, sample_path)
                    baseline_summary = baseline_map[candidate["model_id"]]["roles"][role]
                    dose_record = _elasticity_record(baseline=baseline_summary, adapted=adapted_summary, dose=epoch, training_seconds=cumulative_seconds, training_tokens=cumulative_tokens, trainable_parameters=trainable_parameters, adapter_manifest=adapter_manifest, training_peak_vram_bytes=training_peak, mean_loss=_mean(epoch_losses))
                    role_doses.append(dose_record)
                    _atomic_json(proof_root / "role_results" / model_slug / f"{role}-dose-{epoch}.json", dose_record)
                    print("ELASTICITY_DOSE_COMPLETE_JSON " + json.dumps({"model_id": candidate["model_id"], "role": role, "dose_epoch": epoch, "baseline_quality_100": dose_record["baseline_quality_100"], "adapted_quality_100": dose_record["adapted_quality_100"], "delta_quality_points": dose_record["delta_quality_points"], "training_seconds": cumulative_seconds, "trainable_parameters": trainable_parameters}, sort_keys=True), flush=True)

                model_result["roles"][role] = {"trainable_parameters": trainable_parameters, "target_module_count": len(target_modules), "doses": role_doses}
                base_model = peft_model.unload()
                for parameter in base_model.parameters():
                    parameter.requires_grad_(False)
                gc.collect()
                torch.cuda.empty_cache()

            final_deltas = [model_result["roles"][role]["doses"][-1]["delta_quality_points"] for role in protocol["roles"]]
            final_scores = [model_result["roles"][role]["doses"][-1]["adapted_quality_100"] for role in protocol["roles"]]
            model_result["mean_final_delta_quality_points"] = _mean(final_deltas)
            model_result["mean_final_adapted_quality_100"] = _mean(final_scores)
            model_result["mean_delta_per_gpu_hour"] = _mean([model_result["roles"][role]["doses"][-1]["delta_per_gpu_hour"] for role in protocol["roles"]])
            results.append(model_result)
            _atomic_json(proof_root / "model_results" / f"{model_slug}.json", model_result)
            print("ELASTICITY_MODEL_COMPLETE_JSON " + json.dumps({"model_id": candidate["model_id"], "mean_final_delta_quality_points": model_result["mean_final_delta_quality_points"], "mean_final_adapted_quality_100": model_result["mean_final_adapted_quality_100"], "mean_delta_per_gpu_hour": model_result["mean_delta_per_gpu_hour"]}, sort_keys=True), flush=True)
        except Exception as exc:
            failure = {"model_id": candidate["model_id"], "revision": candidate["revision"], "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
            failures.append(failure)
            _atomic_json(proof_root / "runtime" / f"{model_slug}-failure.json", failure)
            print("ELASTICITY_MODEL_FAILURE_JSON " + json.dumps({k: failure[k] for k in ("model_id", "error_type", "error")}, sort_keys=True), flush=True)
        finally:
            del base_model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    if failures or len(results) != len(topology["candidates"]):
        final = {"result_version": "hephaestus-adaptation-elasticity.v1", "status": "failed", "disposition": "incomplete_evidence", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha, "topology_protocol_sha256": topology_sha, "completed_models": len(results), "expected_models": len(topology["candidates"]), "model_results": results, "failures": failures, "promotion_performed": False, "lineage_mutated": False}
        _atomic_json(proof_root / "elasticity_result.json", final)
        _atomic_json(execution_root / "driver_result.json", final)
        return 2

    role_rankings: dict[str, Any] = {}
    for role in protocol["roles"]:
        rows = []
        for model in results:
            final_dose = model["roles"][role]["doses"][-1]
            first_dose = model["roles"][role]["doses"][0]
            rows.append({"model_id": model["model_id"], "baseline_quality_100": final_dose["baseline_quality_100"], "dose1_quality_100": first_dose["adapted_quality_100"], "dose2_quality_100": final_dose["adapted_quality_100"], "dose1_delta": first_dose["delta_quality_points"], "dose2_delta": final_dose["delta_quality_points"], "dose2_delta_per_gpu_hour": final_dose["delta_per_gpu_hour"], "dose2_delta_per_million_params": final_dose["delta_per_million_trainable_parameters"]})
        role_rankings[role] = {"by_final_quality": sorted(rows, key=lambda row: row["dose2_quality_100"], reverse=True), "by_delta": sorted(rows, key=lambda row: row["dose2_delta"], reverse=True), "by_gpu_hour_elasticity": sorted(rows, key=lambda row: row["dose2_delta_per_gpu_hour"], reverse=True)}

    final = {"result_version": "hephaestus-adaptation-elasticity.v1", "status": "completed", "disposition": "scientific_elasticity_complete", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha, "topology_protocol_sha256": topology_sha, "baseline_run_id": protocol["baseline"]["run_id"], "model_results": results, "role_rankings": role_rankings, "training_performed": True, "promotion_performed": False, "lineage_mutated": False}
    _atomic_json(proof_root / "elasticity_result.json", final)
    _atomic_json(execution_root / "driver_result.json", final)
    print("ADAPTATION_ELASTICITY_RESULT_JSON " + json.dumps({"status": final["status"], "disposition": final["disposition"], "role_rankings": role_rankings}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
