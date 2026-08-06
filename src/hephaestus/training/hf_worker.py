"""Isolated bounded PyTorch/Transformers causal-language-model trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any

_STOP_REASON: str | None = None


class NonFiniteMetricError(RuntimeError):
    pass


class SimulatedOutOfMemoryError(RuntimeError):
    pass


def _request_interrupt(_signum: int, _frame: object) -> None:
    global _STOP_REASON
    if _STOP_REASON is None:
        _STOP_REASON = "interrupted"


def _request_cancel(_signum: int, _frame: object) -> None:
    global _STOP_REASON
    _STOP_REASON = "cancelled"


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _event(root: Path, run_id: str, step: int, status: str, message: str, category: str = "status") -> None:
    _append_jsonl(
        root / "events.jsonl",
        {
            "run_id": run_id,
            "step": step,
            "category": category,
            "status": status,
            "message": message,
            "created_at_unix": time.time(),
        },
    )


def _load_jsonl(path: Path, maximum_rows: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if len(records) >= maximum_rows:
                raise ValueError("processed dataset exceeds prepared row bound")
            record = json.loads(line)
            if not isinstance(record, dict) or not str(record.get("text", "")):
                raise ValueError(f"processed dataset row {line_number} has no text")
            records.append(record)
    if not records:
        raise ValueError("processed dataset is empty")
    return records


def _tokenize_records(
    tokenizer: Any,
    records: list[dict[str, object]],
    config: dict[str, object],
    evidence: dict[str, object],
) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    context_length = int(config["context_length"])
    maximum_tokens = int(config["max_total_tokens"])
    ignored = int(config["ignored_label_token"])
    boundary = evidence.get("prompt_target_boundary", {})
    target_marker = str(boundary.get("target_marker", "<|target|>")) if isinstance(boundary, dict) else "<|target|>"
    eos_id = int(config["special_token_ids"]["eos_token_id"])  # type: ignore[index]
    encoded: list[dict[str, list[int]]] = []
    total_tokens = 0
    truncated = 0
    dropped = 0
    prompt_masked = 0
    for record in records:
        text = str(record["text"])
        record_kind = str(record.get("record_kind", "text"))
        labels: list[int]
        if (
            record_kind == "prompt_target"
            and config.get("prompt_masking") == "mask_prompt_for_prompt_target"
            and target_marker in text
        ):
            prompt, target = text.split(target_marker, 1)
            prompt_ids = tokenizer.encode(prompt + target_marker, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            input_ids = [*prompt_ids, *target_ids]
            labels = [ignored] * len(prompt_ids) + list(target_ids)
            prompt_masked += 1
        else:
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            labels = list(input_ids)
        if not input_ids or input_ids[-1] != eos_id:
            input_ids.append(eos_id)
            labels.append(eos_id)
        if len(input_ids) > context_length:
            input_ids = input_ids[:context_length]
            labels = labels[:context_length]
            truncated += 1
        if not input_ids or all(label == ignored for label in labels):
            dropped += 1
            continue
        total_tokens += len(input_ids)
        if total_tokens > maximum_tokens:
            raise ValueError("tokenized dataset exceeds max_total_tokens")
        encoded.append({"input_ids": list(input_ids), "labels": labels})
    if not encoded:
        raise ValueError("tokenization produced no supervised samples")
    return encoded, {
        "source_samples": len(records),
        "encoded_samples": len(encoded),
        "dropped_samples": dropped,
        "truncated_samples": truncated,
        "prompt_masked_samples": prompt_masked,
        "total_tokens": total_tokens,
    }


def _batches(
    torch: Any,
    samples: list[dict[str, list[int]]],
    batch_size: int,
    pad_id: int,
    ignored: int,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for offset in range(0, len(samples), batch_size):
        group = samples[offset : offset + batch_size]
        width = max(len(sample["input_ids"]) for sample in group)
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for sample in group:
            padding = width - len(sample["input_ids"])
            input_rows.append(sample["input_ids"] + [pad_id] * padding)
            label_rows.append(sample["labels"] + [ignored] * padding)
            attention_rows.append([1] * len(sample["input_ids"]) + [0] * padding)
        batches.append(
            {
                "input_ids": torch.tensor(input_rows, dtype=torch.long),
                "labels": torch.tensor(label_rows, dtype=torch.long),
                "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            }
        )
    return batches


def _loader_kwargs(
    identifier: str, revision: str, config: dict[str, object], *, model_loader: bool
) -> dict[str, object]:
    local = Path(identifier).expanduser().is_dir()
    kwargs: dict[str, object] = {
        "trust_remote_code": False,
        "local_files_only": bool(config.get("local_files_only", True)),
        "token": False,
    }
    if config.get("cache_dir"):
        kwargs["cache_dir"] = str(config["cache_dir"])
    if not local:
        kwargs["revision"] = revision
    loader = config.get("loader_settings", {})
    if model_loader and isinstance(loader, dict):
        for key in ("use_safetensors", "attn_implementation"):
            if key in loader:
                kwargs[key] = loader[key]
    return kwargs


def _validate_loaded_identity(model: Any, tokenizer: Any, config: dict[str, object]) -> None:
    architecture = str(getattr(model.config, "model_type", ""))
    if architecture != config["architecture_family"]:
        raise ValueError(f"architecture mismatch: expected {config['architecture_family']}, observed {architecture}")
    vocabulary_size = int(config["vocabulary_size"])
    if len(tokenizer) != vocabulary_size:
        raise ValueError(f"tokenizer vocabulary mismatch: expected {vocabulary_size}, observed {len(tokenizer)}")
    embedding_size = int(model.get_input_embeddings().weight.shape[0])
    if embedding_size != vocabulary_size:
        raise ValueError(f"model vocabulary mismatch: expected {vocabulary_size}, observed {embedding_size}")
    expected_special = config["special_token_ids"]
    for name in ("eos_token_id", "pad_token_id", "bos_token_id", "unk_token_id"):
        if name in expected_special and getattr(tokenizer, name, None) != expected_special[name]:
            raise ValueError(f"special token mismatch for {name}")
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define distinct explicit EOS and padding IDs")
    model_context = getattr(model.config, "max_position_embeddings", None)
    if model_context is None:
        model_context = getattr(model.config, "n_positions", None)
    if model_context is not None and int(model_context) < int(config["context_length"]):
        raise ValueError("requested context length exceeds model capacity")
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = int(config["parameter_count"])
    if actual_parameters != expected_parameters:
        raise ValueError(f"parameter count mismatch: expected {expected_parameters}, observed {actual_parameters}")


def _component_manifest(checkpoint_partial: Path) -> dict[str, str]:
    return {
        path.relative_to(checkpoint_partial).as_posix(): _hash_file(path)
        for path in sorted(checkpoint_partial.rglob("*"))
        if path.is_file() and path.name != "checkpoint_manifest.json"
    }


def _save_checkpoint(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    job: dict[str, object],
    step: int,
    epoch: float,
    metrics: dict[str, object],
) -> tuple[Path, str]:
    root = Path(str(job["artifact_root"]))
    partial = root / f"checkpoint_step_{step}.partial"
    final = root / f"checkpoint_step_{step}"
    if partial.exists() or final.exists():
        raise RuntimeError(f"checkpoint destination already exists for step {step}")
    partial.mkdir(parents=False, exist_ok=False)
    try:
        model.save_pretrained(partial / "model", safe_serialization=False)
        tokenizer.save_pretrained(partial / "tokenizer")
        state = {
            "step": step,
            "epoch": epoch,
            "examples_processed": int(metrics.get("examples_processed", 0)),
            "tokens_processed": int(metrics.get("tokens_processed", 0)),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": None,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config_fingerprint": job["config_fingerprint"],
            "compatibility": job["compatibility"],
        }
        torch.save(state, partial / "training_state.pt")
        _write_json(
            partial / "loading_instructions.json",
            {
                "backend": "transformers_causal_lm",
                "model_artifact_ref": str(final / "model"),
                "tokenizer_artifact_ref": str(final / "tokenizer"),
                "architecture": job["compatibility"]["architecture_family"],  # type: ignore[index]
                "model_revision": job["compatibility"]["model_revision"],  # type: ignore[index]
                "tokenizer_revision": job["compatibility"]["tokenizer_revision"],  # type: ignore[index]
                "trust_remote_code": False,
                "integrity_manifest_ref": str(final / "checkpoint_manifest.json"),
            },
        )
        components = _component_manifest(partial)
        manifest_hash = _canonical_hash(components)
        _write_json(
            partial / "checkpoint_manifest.json",
            {
                "hash_type": "sha256",
                "components": components,
                "manifest_hash": manifest_hash,
                "partial_write": False,
            },
        )
        os.replace(partial, final)
    except BaseException:
        # A failed staged directory remains explicitly marked partial.  It is
        # never removed or treated as resumable evidence.
        raise
    record = {
        "run_id": job["run_id"],
        "experiment_id": job["experiment_id"],
        "checkpoint_ref": str(final),
        "step": step,
        "epoch": epoch,
        "metric_evidence": metrics,
        "checkpoint_manifest_hash": manifest_hash,
        "hash_type": "sha256_component_manifest",
        "integrity_level": "component_hash_manifest_verified",
        "trainer_ref": "hephaestus.training.hf_worker:v1",
        "config_ref": str(root / "normalized_training_config.json"),
        "model_id": job["compatibility"]["model_id"],  # type: ignore[index]
        "model_revision": job["compatibility"]["model_revision"],  # type: ignore[index]
        "tokenizer_id": job["compatibility"]["tokenizer_id"],  # type: ignore[index]
        "tokenizer_revision": job["compatibility"]["tokenizer_revision"],  # type: ignore[index]
        "data_identity": job["compatibility"]["processed_dataset_hash"],  # type: ignore[index]
        "config_fingerprint": job["config_fingerprint"],
        "resume_compatibility": job["compatibility"],
        "partial_write": False,
        "failure_status": None,
        "generation_handoff_ref": str(final / "loading_instructions.json"),
    }
    _write_json(root / "checkpoint_record.json", record)
    _write_json(
        root / "resume_token.json",
        {
            "run_id": job["run_id"],
            "checkpoint_ref": str(final),
            "checkpoint_manifest_hash": manifest_hash,
            "config_fingerprint": job["config_fingerprint"],
            "compatibility": job["compatibility"],
        },
    )
    return final, manifest_hash


def _make_scheduler(torch: Any, optimizer: Any, kind: str, warmup: int, maximum: int) -> Any:
    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-12, float(step + 1) / float(warmup))
        if kind == "linear":
            remaining = max(0, maximum - step)
            denominator = max(1, maximum - warmup)
            return max(0.0, float(remaining) / float(denominator))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _memory_usage(torch: Any, device: str) -> dict[str, int | None]:
    if device == "cuda":
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1024 if sys.platform != "darwin" else 1
    return {"process_max_rss_bytes": int(maximum_rss * multiplier)}


def run(job_path: Path, resume_token_path: Path | None = None) -> int:
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    job = json.loads(job_path.read_text(encoding="utf-8"))
    config = dict(job["training_config"])
    compatibility = dict(job["compatibility"])
    root = Path(job["artifact_root"])
    run_id = str(job["run_id"])
    device = str(config["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if config.get("simulate_oom") in {"cpu", "gpu"}:
        raise SimulatedOutOfMemoryError(f"simulated_{config['simulate_oom']}_oom")

    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
    deterministic_warning: str | None = None
    try:
        torch.use_deterministic_algorithms(True)
    except (RuntimeError, AttributeError) as exc:
        deterministic_warning = f"deterministic algorithm request unavailable: {exc}"

    resume_state: dict[str, object] | None = None
    if resume_token_path is not None:
        token = json.loads(resume_token_path.read_text(encoding="utf-8"))
        checkpoint = Path(str(token["checkpoint_ref"]))
        model_source = str(checkpoint / "model")
        tokenizer_source = str(checkpoint / "tokenizer")
        model_kwargs = {"trust_remote_code": False, "local_files_only": True, "token": False}
        tokenizer_kwargs = dict(model_kwargs)
        resume_state = torch.load(checkpoint / "training_state.pt", map_location="cpu")
        if resume_state.get("config_fingerprint") != job["config_fingerprint"]:
            raise ValueError("checkpoint config fingerprint mismatch")
        if resume_state.get("compatibility") != compatibility:
            raise ValueError("checkpoint compatibility mismatch")
    else:
        model_source = compatibility["model_id"]
        tokenizer_source = compatibility["tokenizer_id"]
        model_kwargs = _loader_kwargs(
            model_source, compatibility["model_revision"], config, model_loader=True
        )
        tokenizer_kwargs = _loader_kwargs(
            tokenizer_source, compatibility["tokenizer_revision"], config, model_loader=False
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    _validate_loaded_identity(model, tokenizer, config)
    model.to(device=device, dtype=getattr(torch, str(config["dtype"])))
    model.train()

    evidence_ref = Path(str(config["processing_evidence_ref"]))
    evidence = json.loads(evidence_ref.read_text(encoding="utf-8"))
    records = _load_jsonl(Path(compatibility["processed_dataset_ref"]), int(config["maximum_rows"]))
    samples, tokenization_summary = _tokenize_records(tokenizer, records, config, evidence)
    batches = _batches(
        torch,
        samples,
        int(config["batch_size"]),
        int(config["special_token_ids"]["pad_token_id"]),
        int(config["ignored_label_token"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = _make_scheduler(
        torch,
        optimizer,
        str(config["scheduler"]),
        int(config["warmup_steps"]),
        int(config["max_steps"]),
    )
    start_step = 0
    examples_processed = 0
    tokens_processed = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        random.setstate(resume_state["python_rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state"])
        start_step = int(resume_state["step"])
        examples_processed = int(resume_state.get("examples_processed", 0))
        tokens_processed = int(resume_state.get("tokens_processed", 0))
        _event(root, run_id, start_step, "resuming", "strict checkpoint state loaded")

    _write_json(root / "tokenization_summary.json", tokenization_summary)
    _event(root, run_id, start_step, "running", "bounded causal-LM fine-tuning started")
    start_time = time.monotonic()
    last_loss = math.inf
    last_grad_norm = math.inf
    latest_checkpoint: Path | None = None
    accumulation = int(config["gradient_accumulation_steps"])
    maximum_steps = int(config["max_steps"])
    if config.get("max_epochs") is not None:
        steps_per_epoch = max(1, math.ceil(len(batches) / accumulation))
        epoch_steps = math.ceil(float(config["max_epochs"]) * steps_per_epoch)
        maximum_steps = min(maximum_steps, epoch_steps)
    checkpoint_every = int(config["checkpoint_every_steps"])
    log_every = int(config["logging_every_steps"])
    delay = float(config.get("step_delay_seconds", 0.0))

    for step in range(start_step + 1, maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        step_tokens = 0
        step_examples = 0
        for micro_step in range(accumulation):
            index = ((step - 1) * accumulation + micro_step) % len(batches)
            batch = {name: tensor.to(device) for name, tensor in batches[index].items()}
            output = model(**batch)
            loss = output.loss
            if not torch.isfinite(loss):
                raise NonFiniteMetricError(f"non-finite training loss at step {step}")
            (loss / accumulation).backward()
            accumulated_loss += float(loss.detach().cpu())
            step_tokens += int(batch["attention_mask"].sum().item())
            step_examples += int(batch["input_ids"].shape[0])
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clipping"]))
        if not torch.isfinite(grad_norm):
            raise NonFiniteMetricError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        last_loss = accumulated_loss / accumulation
        last_grad_norm = float(grad_norm.detach().cpu())
        tokens_processed += step_tokens
        examples_processed += step_examples
        elapsed = max(time.monotonic() - start_time, 1e-9)
        epoch = examples_processed / max(1, len(samples))
        metric = {
            "run_id": run_id,
            "step": step,
            "epoch": epoch,
            "training_loss": last_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm": last_grad_norm,
            "tokens_processed": tokens_processed,
            "examples_processed": examples_processed,
            "tokens_per_second": tokens_processed / elapsed,
            "examples_per_second": examples_processed / elapsed,
            "elapsed_seconds": elapsed,
            "memory": _memory_usage(torch, device),
        }
        if step % log_every == 0:
            _append_jsonl(root / "metrics.jsonl", metric)
        if step % checkpoint_every == 0:
            latest_checkpoint, _manifest_hash = _save_checkpoint(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                job=job,
                step=step,
                epoch=epoch,
                metrics=metric,
            )
        if delay:
            time.sleep(delay)
        if _STOP_REASON == "cancelled":
            _event(root, run_id, step, "cancelled", "graceful cancellation")
            _write_json(root / "runtime_result.json", {"status": "cancelled", "step": step})
            return 143
        if _STOP_REASON == "interrupted":
            if latest_checkpoint is None or latest_checkpoint.name != f"checkpoint_step_{step}":
                latest_checkpoint, _manifest_hash = _save_checkpoint(
                    torch=torch,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    job=job,
                    step=step,
                    epoch=epoch,
                    metrics=metric,
                )
            _event(root, run_id, step, "interrupted", "graceful resumable interruption")
            _write_json(
                root / "runtime_result.json",
                {"status": "interrupted", "step": step, "checkpoint_ref": str(latest_checkpoint)},
            )
            return 130

    if latest_checkpoint is None or latest_checkpoint.name != f"checkpoint_step_{maximum_steps}":
        epoch = examples_processed / max(1, len(samples))
        latest_checkpoint, manifest_hash = _save_checkpoint(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            job=job,
            step=maximum_steps,
            epoch=epoch,
            metrics={"training_loss": last_loss, "gradient_norm": last_grad_norm},
        )
    else:
        manifest_hash = json.loads((latest_checkpoint / "checkpoint_manifest.json").read_text())["manifest_hash"]
    elapsed = max(time.monotonic() - start_time, 1e-9)
    summary = {
        "run_id": run_id,
        "steps": maximum_steps,
        "final_training_loss": last_loss,
        "final_gradient_norm": last_grad_norm,
        "tokens_processed": tokens_processed,
        "examples_processed": examples_processed,
        "tokens_per_second": tokens_processed / elapsed,
        "elapsed_seconds": elapsed,
        "finite": math.isfinite(last_loss) and math.isfinite(last_grad_norm),
        "memory": _memory_usage(torch, device),
        "determinism": {
            "seed": config["seed"],
            "deterministic_algorithms_requested": True,
            "warning": deterministic_warning,
            "perfect_determinism_claimed": False,
        },
        "limitation": "Training loss is optimization evidence, not semantic evaluation evidence.",
    }
    _write_json(root / "metrics_summary.json", summary)
    final_result = {
        "status": "completed",
        "checkpoint_ref": str(latest_checkpoint),
        "checkpoint_manifest_hash": manifest_hash,
        "metrics_ref": str(root / "metrics_summary.json"),
        "model_artifact_ref": str(latest_checkpoint / "model"),
        "tokenizer_artifact_ref": str(latest_checkpoint / "tokenizer"),
        "generation_handoff_ref": str(latest_checkpoint / "loading_instructions.json"),
    }
    _write_json(root / "final_result.json", final_result)
    _event(root, run_id, maximum_steps, "completed", "bounded causal-LM fine-tuning completed")
    _write_json(root / "runtime_result.json", final_result)
    return 0


def _normalize_error(exc: BaseException, job: dict[str, object]) -> dict[str, object]:
    config = job.get("training_config", {})
    device = str(config.get("device", "unknown")) if isinstance(config, dict) else "unknown"
    message = f"{type(exc).__name__}: {exc}"
    lowered = message.lower()
    error_type = "runtime_failure"
    if (
        isinstance(exc, MemoryError)
        or "out of memory" in lowered
        or "simulated_cpu_oom" in lowered
        or "simulated_gpu_oom" in lowered
    ):
        error_type = "gpu_oom" if device == "cuda" or "simulated_gpu_oom" in lowered else "cpu_oom"
    if isinstance(exc, NonFiniteMetricError):
        error_type = "non_finite_metric"
    return {
        "status": "failed",
        "error_type": error_type,
        "error": message,
        "device": device,
        "batch_size": config.get("batch_size") if isinstance(config, dict) else None,
        "context_length": config.get("context_length") if isinstance(config, dict) else None,
        "suggested_reductions": [
            "reduce batch_size",
            "reduce context_length",
            "enable gradient accumulation while preserving effective batch intent",
        ] if error_type in {"cpu_oom", "gpu_oom"} else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--resume-token", type=Path)
    arguments = parser.parse_args()
    signal.signal(signal.SIGINT, _request_interrupt)
    signal.signal(signal.SIGTERM, _request_cancel)
    job: dict[str, object] = {}
    try:
        job = json.loads(arguments.job.read_text(encoding="utf-8"))
        return run(arguments.job, arguments.resume_token)
    except BaseException as exc:  # noqa: BLE001 - process boundary normalizes unknown failures
        root = Path(str(job.get("artifact_root", arguments.job.parent)))
        root.mkdir(parents=True, exist_ok=True)
        result = _normalize_error(exc, job)
        _write_json(root / "runtime_result.json", result)
        _event(root, str(job.get("run_id", "unknown")), 0, "failed", str(result["error_type"]), "incident")
        print(f"Transformers worker failed: {result['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
