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
import adaptation_elasticity_resume_v1 as resume
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


def _write_once_json(path: Path, payload: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite persisted scientific evidence: {path}")
    _atomic_json(path, payload)


def _preserve_or_initialize_json(path: Path, payload: object, *, label: str) -> None:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise RuntimeError(f"persisted {label} differs from the frozen protocol")
        return
    _write_once_json(path, payload)


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
        hashes[path.relative_to(root).as_posix()] = "sha256:" + resume.sha256_file(path)
        total += path.stat().st_size
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


def _adapter_manifest(
    path: Path,
    *,
    model_id: str,
    revision: str,
    role: str,
    dose: int,
    trainable_parameters: int,
    target_modules: list[str],
    evidence_identity: dict[str, str],
) -> dict[str, Any]:
    hashes, byte_size = _hash_files(path)
    return {
        "manifest_version": resume.CURRENT_MANIFEST_VERSION,
        "model_id": model_id,
        "revision": revision,
        "role": role,
        "dose_epoch": dose,
        "trainable_parameters": trainable_parameters,
        "target_module_count": len(target_modules),
        "target_modules_sha256": _sha("\n".join(target_modules).encode()),
        "components": hashes,
        "adapter_bytes": byte_size,
        "manifest_sha256": _sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()),
        **evidence_identity,
    }


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


def _elasticity_record(
    *,
    baseline: dict[str, Any],
    adapted: dict[str, Any],
    dose: int,
    training_seconds: float,
    training_tokens: int,
    trainable_parameters: int,
    adapter_manifest: dict[str, Any],
    training_peak_vram_bytes: int,
    mean_loss: float,
    evidence_identity: dict[str, str] | None = None,
    sample_path: Path | None = None,
) -> dict[str, Any]:
    delta = float(adapted["quality_100"]) - float(baseline["quality_100"])
    gpu_hours = training_seconds / 3600.0
    params_m = trainable_parameters / 1_000_000.0
    tokens_m = training_tokens / 1_000_000.0
    record = {"dose_epoch": dose, "baseline_quality_100": baseline["quality_100"], "adapted_quality_100": adapted["quality_100"], "delta_quality_points": delta, "delta_per_gpu_hour": delta / gpu_hours if gpu_hours > 0 else None, "delta_per_million_trainable_parameters": delta / params_m if params_m > 0 else None, "delta_per_million_training_tokens": delta / tokens_m if tokens_m > 0 else None, "schema_compliance_delta": adapted["schema_compliance"] - baseline["schema_compliance"], "evidence_grounding_delta": adapted["evidence_grounding"] - baseline["evidence_grounding"], "confidence_calibration_delta": adapted["confidence_calibration"] - baseline["confidence_calibration"], "hallucination_rate_delta": adapted["hallucination_rate"] - baseline["hallucination_rate"], "training_seconds_cumulative": training_seconds, "gpu_hours_cumulative": gpu_hours, "training_tokens_cumulative": training_tokens, "trainable_parameters": trainable_parameters, "adapter_bytes": adapter_manifest["adapter_bytes"], "training_peak_vram_bytes": training_peak_vram_bytes, "mean_training_loss_epoch": mean_loss, "adapted_role_summary": adapted, "adapter_manifest": adapter_manifest}
    if evidence_identity is not None:
        record["evidence_identity"] = evidence_identity
    if sample_path is not None:
        record["post_samples"] = {
            "sample_count": adapted["sample_count"],
            "sha256": "sha256:" + resume.sha256_file(sample_path),
        }
    return record


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _evidence_identity(
    protocol: dict[str, Any],
    protocol_sha: str,
    topology_sha: str,
    admission: dict[str, Any],
) -> dict[str, str]:
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "topology_protocol_sha256": topology_sha,
        "dataset_sha256": admission["dataset_manifest"]["dataset_sha256"],
        "training_config_sha256": resume.sha256_json(protocol["training"]),
    }


def _validate_or_initialize_run(
    proof_root: Path,
    *,
    protocol: dict[str, Any],
    protocol_sha: str,
    topology: dict[str, Any],
    topology_sha: str,
    admission: dict[str, Any],
) -> dict[str, Any]:
    run_id = _required("HEPHAESTUS_PROOF_RUN_ID")
    current_repo_sha = _required("HEPHAESTUS_REPO_SHA")
    run_manifest_path = proof_root / "run_manifest.json"
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        expected = {
            "run_id": run_id,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "topology_protocol_sha256": topology_sha,
            "baseline_run_id": protocol["baseline"]["run_id"],
            "candidate_order": [row["model_id"] for row in topology["candidates"]],
            "roles": protocol["roles"],
            "training_method": "lora",
            "promotion_allowed": False,
            "lineage_mutation_allowed": False,
        }
        for key, value in expected.items():
            if run_manifest.get(key) != value:
                raise RuntimeError(f"persisted run manifest mismatch: {key}")
        origin_repo_sha = str(run_manifest.get("repo_sha") or "")
        if not origin_repo_sha:
            raise RuntimeError("persisted run manifest lacks its origin repository SHA")
    else:
        origin_repo_sha = current_repo_sha
        run_manifest = {
            "run_id": run_id,
            "repo_sha": origin_repo_sha,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "topology_protocol_sha256": topology_sha,
            "baseline_run_id": protocol["baseline"]["run_id"],
            "candidate_order": [row["model_id"] for row in topology["candidates"]],
            "roles": protocol["roles"],
            "training_method": "lora",
            "promotion_allowed": False,
            "lineage_mutation_allowed": False,
            "started_at_unix": time.time(),
        }
        _write_once_json(run_manifest_path, run_manifest)

    _preserve_or_initialize_json(proof_root / "protocol_snapshot.json", protocol, label="protocol snapshot")
    _preserve_or_initialize_json(
        proof_root / "topology_protocol_snapshot.json",
        topology,
        label="topology protocol snapshot",
    )

    origin_admission_root = SCIENTIFIC_ROOT / "adaptation_elasticity" / "model_admission" / origin_repo_sha
    origin_admission = json.loads((origin_admission_root / "admission.json").read_text(encoding="utf-8"))
    if origin_admission.get("repo_sha") != origin_repo_sha:
        raise RuntimeError("origin admission repository SHA mismatch")
    for key, expected in (
        ("protocol_sha256", protocol_sha),
        ("topology_protocol_sha256", topology_sha),
        ("candidate_order", [row["model_id"] for row in topology["candidates"]]),
    ):
        if origin_admission.get(key) != expected:
            raise RuntimeError(f"origin admission mismatch: {key}")
    origin_dataset = (origin_admission_root / "training.jsonl").read_bytes()
    origin_dataset_sha = _sha(origin_dataset)
    current_dataset_sha = admission["dataset_manifest"]["dataset_sha256"]
    if origin_dataset_sha != current_dataset_sha or origin_admission["dataset_manifest"]["dataset_sha256"] != current_dataset_sha:
        raise RuntimeError("resume dataset bytes differ from the original immutable dataset")
    contamination = json.loads((origin_admission_root / "contamination_report.json").read_text(encoding="utf-8"))
    if contamination.get("passed") is not True or contamination.get("dataset_sha256") != current_dataset_sha:
        raise RuntimeError("origin contamination evidence is invalid")
    origin_requirements = (origin_admission_root / "requirements.txt").read_bytes()
    if _sha(origin_requirements) != origin_admission.get("requirements_sha256"):
        raise RuntimeError("origin dependency lock hash mismatch")
    return {
        "run_id": run_id,
        "origin_repo_sha": origin_repo_sha,
        "current_repo_sha": current_repo_sha,
        "resumed": run_manifest_path.exists() and origin_repo_sha != current_repo_sha,
        "dataset_sha256": current_dataset_sha,
        "origin_admission_root": str(origin_admission_root),
    }


def _model_context_errors(
    proof_root: Path,
    candidate: dict[str, Any],
    admission: dict[str, Any],
    protocol: dict[str, Any],
) -> list[str]:
    model_slug = _slug(candidate["model_id"])
    location = resume.standard_location(proof_root, model_slug)
    has_persisted_role_evidence = any(
        location.result_path(role, dose).exists()
        or location.adapter_path(role, dose).exists()
        or location.sample_path(role, dose).exists()
        for role in protocol["roles"]
        for dose in protocol["training"]["dose_checkpoints"]
    )
    if not has_persisted_role_evidence:
        return []
    errors: list[str] = []
    path = proof_root / "model_manifests" / model_slug / "snapshot_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"persisted base-model manifest is unavailable: {type(exc).__name__}: {exc}"]
    for key, expected in (
        ("model_id", candidate["model_id"]),
        ("revision", candidate["revision"]),
        ("license", candidate["license"]),
        ("trust_remote_code", False),
        ("cache_persistence", "ephemeral"),
    ):
        if manifest.get(key) != expected:
            errors.append(f"persisted base-model manifest mismatch: {key}")
    components = manifest.get("components")
    if not isinstance(components, dict):
        errors.append("persisted base-model manifest components are invalid")
        components = {}
    expected_manifest_hash = "sha256:" + _sha(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    )
    if manifest.get("manifest_hash") != expected_manifest_hash:
        errors.append("persisted base-model component manifest hash mismatch")
    admitted = next(
        (
            row
            for row in admission["candidates"]
            if row["model_id"] == candidate["model_id"] and row["revision"] == candidate["revision"]
        ),
        None,
    )
    if admitted is None:
        errors.append("persisted model is absent from current immutable admission")
    else:
        for name, expected in admitted["metadata"]["metadata_component_hashes"].items():
            if components.get(name) != expected:
                errors.append(f"persisted base-model metadata hash mismatch: {name}")
    return errors


def _summarize_role(
    topology: dict[str, Any],
    candidate: dict[str, Any],
    role: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    role_spec = dict(topology)
    role_spec["roles"] = [role]
    return topology_runner._summarize_model(role_spec, candidate, samples, {})["roles"][role]


def _select_role(
    proof_root: Path,
    *,
    candidate: dict[str, Any],
    role: str,
    protocol: dict[str, Any],
    topology: dict[str, Any],
    baseline_summary: dict[str, Any],
    evidence_identity: dict[str, str],
    context_errors: list[str],
) -> dict[str, Any]:
    model_slug = _slug(candidate["model_id"])
    selection = resume.choose_role_evidence(
        proof_root,
        model_slug=model_slug,
        candidate=candidate,
        role=role,
        protocol=protocol,
        topology=topology,
        baseline_summary=baseline_summary,
        identity=evidence_identity,
        score_response=topology_runner.score_response,
        summarize_role=lambda samples: _summarize_role(topology, candidate, role, samples),
    )
    selection["context_errors"] = list(context_errors)
    if not context_errors:
        return selection
    complete_reconstructions = [
        row for row in selection["reconstructions"] if row["state"] == "complete"
    ]
    if complete_reconstructions:
        selection["state"] = "complete"
        selection["selected"] = complete_reconstructions[0]
    elif selection["standard"]["state"] != "missing":
        selection["state"] = "invalid"
        selection["selected"] = None
    return selection


def _role_payload(inspection: dict[str, Any]) -> dict[str, Any]:
    doses = [dose["record"] for dose in inspection["doses"]]
    if any(not isinstance(dose, dict) for dose in doses):
        raise RuntimeError("cannot build a role result from incomplete dose evidence")
    manifest = doses[0]["adapter_manifest"]
    return {
        "trainable_parameters": manifest["trainable_parameters"],
        "target_module_count": manifest["target_module_count"],
        "doses": doses,
        "evidence_selection": {
            "evidence_id": inspection["location"]["evidence_id"],
            "kind": inspection["location"]["kind"],
            "dose_hashes": [dose["evidence_hashes"] for dose in inspection["doses"]],
        },
    }


def _build_model_result(
    candidate: dict[str, Any],
    protocol: dict[str, Any],
    role_results: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    model_result = {
        "model_id": candidate["model_id"],
        "revision": candidate["revision"],
        "license": candidate["license"],
        "roles": role_results,
        "runtime": runtime,
        "status": "complete",
    }
    final_deltas = [role_results[role]["doses"][-1]["delta_quality_points"] for role in protocol["roles"]]
    final_scores = [role_results[role]["doses"][-1]["adapted_quality_100"] for role in protocol["roles"]]
    model_result["mean_final_delta_quality_points"] = _mean(final_deltas)
    model_result["mean_final_adapted_quality_100"] = _mean(final_scores)
    model_result["mean_delta_per_gpu_hour"] = _mean(
        [role_results[role]["doses"][-1]["delta_per_gpu_hour"] for role in protocol["roles"]]
    )
    return model_result


def _output_hashes(dose: dict[str, Any]) -> dict[str, str]:
    return {
        f"{row['case_id']}:{row['seed']}": _sha(row["output"].encode())
        for row in dose["samples"]
    }


def _reconstruction_comparison(
    prior: dict[str, Any] | None,
    recreated: dict[str, Any],
) -> dict[str, Any]:
    if prior is None or prior.get("state") != "complete":
        return {
            "comparison_version": "adaptation-elasticity-reconstruction.v1",
            "disposition": "no_valid_prior_dose_to_compare",
            "matched": None,
        }
    prior_record = prior["record"]
    recreated_record = recreated["record"]
    scientific_summary_keys = (
        "quality",
        "quality_100",
        "schema_compliance",
        "evidence_grounding",
        "confidence_calibration",
        "hallucination_rate",
        "exact_repeatability",
        "decision_repeatability",
        "robustness_to_irrelevant_evidence",
        "contradictory_evidence_quality",
        "sample_count",
        "case_count",
    )
    prior_summary = {key: prior_record["adapted_role_summary"].get(key) for key in scientific_summary_keys}
    recreated_summary = {key: recreated_record["adapted_role_summary"].get(key) for key in scientific_summary_keys}
    checks = {
        "training_tokens_exact": prior_record["training_tokens_cumulative"] == recreated_record["training_tokens_cumulative"],
        "trainable_parameters_exact": prior_record["trainable_parameters"] == recreated_record["trainable_parameters"],
        "sample_outputs_exact": _output_hashes(prior) == _output_hashes(recreated),
        "scored_role_metrics_exact": prior_summary == recreated_summary,
        "mean_training_loss_within_1e-4": math.isclose(
            float(prior_record["mean_training_loss_epoch"]),
            float(recreated_record["mean_training_loss_epoch"]),
            rel_tol=0.0,
            abs_tol=1e-4,
        ),
    }
    prior_weights = prior["evidence_hashes"].get("adapter_components", {}).get("adapter_model.safetensors")
    recreated_weights = recreated["evidence_hashes"].get("adapter_components", {}).get("adapter_model.safetensors")
    matched = all(checks.values())
    return {
        "comparison_version": "adaptation-elasticity-reconstruction.v1",
        "contract": {
            "exact_training_tokens": True,
            "exact_trainable_parameters": True,
            "exact_generated_outputs_by_case_and_seed": True,
            "exact_scored_role_metrics": True,
            "mean_training_loss_absolute_tolerance": 1e-4,
            "bitwise_adapter_equality_required": False,
        },
        "checks": checks,
        "matched": matched,
        "disposition": "deterministic_reconstruction_matched" if matched else "reconstruction_divergence_preserved",
        "prior_evidence_hashes": prior["evidence_hashes"],
        "recreated_evidence_hashes": recreated["evidence_hashes"],
        "adapter_weight_hash_equal": prior_weights == recreated_weights,
    }


def _train_role(
    *,
    base_model: Any,
    tokenizer: Any,
    runtime: dict[str, Any],
    candidate: dict[str, Any],
    role: str,
    role_index: int,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    topology: dict[str, Any],
    baseline_summary: dict[str, Any],
    target_modules: list[str],
    location: resume.RoleEvidenceLocation,
    evidence_identity: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import get_linear_schedule_with_warmup

    train = protocol["training"]
    torch.manual_seed(int(train["model_seed"]) + role_index)
    torch.cuda.manual_seed_all(int(train["model_seed"]) + role_index)
    shuffled = list(rows)
    random.Random(int(train["shuffle_seed"]) + role_index).shuffle(shuffled)
    encoded = [_tokenize_example(tokenizer, row, int(train["max_seq_length"])) for row in shuffled]
    lora_config = LoraConfig(
        r=int(train["rank"]),
        lora_alpha=int(train["alpha"]),
        lora_dropout=float(train["dropout"]),
        bias=str(train["bias"]),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    peft_model = get_peft_model(base_model, lora_config)
    trainable_parameters = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if trainable_parameters <= 0:
        raise RuntimeError(f"no trainable LoRA parameters for {candidate['model_id']} role={role}")
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]),
    )
    accumulation = int(train["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(encoded) / accumulation)
    total_updates = updates_per_epoch * int(train["epochs"])
    warmup_steps = max(1, int(round(total_updates * float(train["warmup_ratio"]))))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    cumulative_seconds = 0.0
    cumulative_tokens = 0
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
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_model.parameters() if p.requires_grad],
                    float(train["max_grad_norm"]),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        cumulative_seconds += time.perf_counter() - epoch_started
        training_peak = int(torch.cuda.max_memory_allocated())
        adapter_dir = location.adapter_path(role, epoch)
        if adapter_dir.exists():
            raise RuntimeError(f"refusing to overwrite adapter evidence: {adapter_dir}")
        peft_model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter_manifest = _adapter_manifest(
            adapter_dir,
            model_id=candidate["model_id"],
            revision=candidate["revision"],
            role=role,
            dose=epoch,
            trainable_parameters=trainable_parameters,
            target_modules=target_modules,
            evidence_identity=evidence_identity,
        )
        _write_once_json(adapter_dir / "hephaestus_adapter_manifest.json", adapter_manifest)
        sample_path = location.sample_path(role, epoch)
        if sample_path.exists():
            raise RuntimeError(f"refusing to overwrite post-sample evidence: {sample_path}")
        adapted_summary = _evaluate_role(
            peft_model,
            tokenizer,
            topology,
            candidate,
            role,
            runtime,
            sample_path,
        )
        dose_record = _elasticity_record(
            baseline=baseline_summary,
            adapted=adapted_summary,
            dose=epoch,
            training_seconds=cumulative_seconds,
            training_tokens=cumulative_tokens,
            trainable_parameters=trainable_parameters,
            adapter_manifest=adapter_manifest,
            training_peak_vram_bytes=training_peak,
            mean_loss=_mean(epoch_losses),
            evidence_identity=evidence_identity,
            sample_path=sample_path,
        )
        _write_once_json(location.result_path(role, epoch), dose_record)
        print(
            "ELASTICITY_DOSE_COMPLETE_JSON "
            + json.dumps(
                {
                    "model_id": candidate["model_id"],
                    "role": role,
                    "dose_epoch": epoch,
                    "baseline_quality_100": dose_record["baseline_quality_100"],
                    "adapted_quality_100": dose_record["adapted_quality_100"],
                    "delta_quality_points": dose_record["delta_quality_points"],
                    "training_seconds": cumulative_seconds,
                    "trainable_parameters": trainable_parameters,
                    "evidence_id": location.evidence_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    base_model = peft_model.unload()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    return base_model, {
        "trainable_parameters": trainable_parameters,
        "target_module_count": len(target_modules),
    }


def _compact_inspection(inspection: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": inspection["state"],
        "valid_doses": inspection.get("valid_doses", []),
        "location": inspection["location"],
        "cross_dose_errors": inspection.get("cross_dose_errors", []),
        "doses": [
            {
                "dose_epoch": dose["dose_epoch"],
                "state": dose["state"],
                "errors": dose["errors"],
                "evidence_hashes": dose["evidence_hashes"],
            }
            for dose in inspection["doses"]
        ],
    }


def _compact_selection(selection: dict[str, Any], action: dict[str, str]) -> dict[str, Any]:
    return {
        "state": selection["state"],
        "action": action["action"],
        "reason": action["reason"],
        "context_errors": selection.get("context_errors", []),
        "selected_evidence_id": (
            selection["selected"]["location"]["evidence_id"]
            if selection.get("selected") is not None
            else None
        ),
        "standard": _compact_inspection(selection["standard"]),
        "reconstructions": [
            _compact_inspection(row) for row in selection.get("reconstructions", [])
        ],
    }


def main() -> int:
    import torch

    protocol, protocol_sha, topology, topology_sha = _load_protocols()
    if protocol["baseline"]["protocol_sha256"] != topology_sha:
        raise RuntimeError("frozen topology bytes changed after adaptation protocol was admitted")
    admission, dataset = _load_admission(protocol_sha, topology_sha)
    baseline = _load_baseline(protocol, topology_sha)
    proof_root, execution_root = _paths()
    proof_root.mkdir(parents=True, exist_ok=True)
    execution_root.mkdir(parents=True, exist_ok=True)
    attempt = int(_required("HEPHAESTUS_ATTEMPT"))
    attempt_root = proof_root / "resume_attempts" / f"attempt-{attempt}"
    if attempt_root.exists():
        raise RuntimeError(f"resume attempt already exists; refusing to overwrite: {attempt_root}")
    attempt_root.mkdir(parents=True)
    run_identity = _validate_or_initialize_run(
        proof_root,
        protocol=protocol,
        protocol_sha=protocol_sha,
        topology=topology,
        topology_sha=topology_sha,
        admission=admission,
    )
    evidence_identity = _evidence_identity(protocol, protocol_sha, topology_sha, admission)

    baseline_map = {row["model_id"]: row for row in baseline["model_summaries"]}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    per_role = {role: [row for row in dataset if row["role"] == role] for role in protocol["roles"]}
    for role, rows in per_role.items():
        if len(rows) != int(protocol["dataset"]["examples_per_role"]):
            raise RuntimeError(f"training dataset count drifted for {role}")

    planned: dict[str, dict[str, Any]] = {}
    plan_record: dict[str, Any] = {
        "resume_plan_version": "adaptation-elasticity-resume-plan.v1",
        "run_identity": run_identity,
        "attempt": attempt,
        "evidence_identity": evidence_identity,
        "models": {},
        "promotion_allowed": False,
        "lineage_mutation_allowed": False,
    }
    for candidate in topology["candidates"]:
        context_errors = _model_context_errors(proof_root, candidate, admission, protocol)
        candidate_plan: dict[str, Any] = {}
        for role in protocol["roles"]:
            selection = _select_role(
                proof_root,
                candidate=candidate,
                role=role,
                protocol=protocol,
                topology=topology,
                baseline_summary=baseline_map[candidate["model_id"]]["roles"][role],
                evidence_identity=evidence_identity,
                context_errors=context_errors,
            )
            action = resume.role_action(selection)
            candidate_plan[role] = {"selection": selection, "action": action}
        planned[candidate["model_id"]] = candidate_plan
        plan_record["models"][candidate["model_id"]] = {
            role: _compact_selection(row["selection"], row["action"])
            for role, row in candidate_plan.items()
        }
    _write_once_json(attempt_root / "resume_plan.json", plan_record)
    print(
        "ADAPTATION_ELASTICITY_RESUME_PLAN_JSON "
        + json.dumps(
            {
                model_id: {
                    role: row["action"]["action"]
                    for role, row in role_plan.items()
                }
                for model_id, role_plan in planned.items()
            },
            sort_keys=True,
        ),
        flush=True,
    )

    audit: dict[str, Any] = {
        "resume_audit_version": "adaptation-elasticity-resume-audit.v1",
        "run_identity": run_identity,
        "attempt": attempt,
        "models": {},
    }

    for candidate in topology["candidates"]:
        base_model = tokenizer = None
        model_slug = _slug(candidate["model_id"])
        role_plan = planned[candidate["model_id"]]
        model_audit: dict[str, Any] = {"roles": {}}
        audit["models"][candidate["model_id"]] = model_audit
        try:
            actions = [role_plan[role]["action"]["action"] for role in protocol["roles"]]
            needs_training = any(action != "reuse" for action in actions)
            target_modules: list[str] = []
            if needs_training:
                snapshot, model_manifest = topology_runner._materialize(candidate, admission, attempt_root)
                base_model, tokenizer, runtime = _load_training_base(snapshot, candidate, protocol)
                runtime.update(
                    {
                        "model_id": candidate["model_id"],
                        "revision": candidate["revision"],
                        "manifest_hash": model_manifest["manifest_hash"],
                        "resume_attempt": attempt,
                    }
                )
                target_modules = _target_modules(
                    base_model,
                    list(protocol["training"]["target_module_suffixes"]),
                )
                runtime["lora_target_module_count"] = len(target_modules)
                runtime["lora_target_modules_sha256"] = _sha("\n".join(target_modules).encode())
                _write_once_json(attempt_root / "runtime" / f"{model_slug}.json", runtime)
                for role in protocol["roles"]:
                    selected = role_plan[role]["selection"].get("selected")
                    if selected is None:
                        continue
                    prior_manifest = selected["doses"][0]["record"]["adapter_manifest"]
                    if prior_manifest["target_module_count"] != len(target_modules):
                        raise RuntimeError("current target-module count differs from reused evidence")
                    if prior_manifest["target_modules_sha256"] != _sha("\n".join(target_modules).encode()):
                        raise RuntimeError("current target-module identity differs from reused evidence")
            else:
                prior_runtime_path = proof_root / "runtime" / f"{model_slug}.json"
                if prior_runtime_path.is_file():
                    runtime = json.loads(prior_runtime_path.read_text(encoding="utf-8"))
                    if runtime.get("model_id") != candidate["model_id"] or runtime.get("revision") != candidate["revision"]:
                        raise RuntimeError("persisted runtime identity mismatch")
                else:
                    runtime = {
                        "model_id": candidate["model_id"],
                        "revision": candidate["revision"],
                        "evidence_reused": True,
                        "runtime_record_unavailable": True,
                    }

            role_results: dict[str, Any] = {}

            for role_index, role in enumerate(protocol["roles"]):
                selection = role_plan[role]["selection"]
                action_record = role_plan[role]["action"]
                action = action_record["action"]
                if action == "reuse":
                    inspection = selection["selected"]
                    assert inspection is not None
                    role_results[role] = _role_payload(inspection)
                    model_audit["roles"][role] = {
                        "action": "reused_persisted_evidence",
                        "reason": action_record["reason"],
                        "selected": _compact_inspection(inspection),
                    }
                    continue

                if base_model is None or tokenizer is None:
                    raise RuntimeError("training was planned without a loaded immutable base model")
                location = resume.output_location_for_action(
                    proof_root,
                    model_slug=model_slug,
                    role=role,
                    attempt=attempt,
                    action=action,
                )
                prior_dose_one = next(
                    (
                        dose
                        for dose in selection["standard"]["doses"]
                        if dose["dose_epoch"] == 1 and dose["state"] == "complete"
                    ),
                    None,
                )
                base_model, _ = _train_role(
                    base_model=base_model,
                    tokenizer=tokenizer,
                    runtime=runtime,
                    candidate=candidate,
                    role=role,
                    role_index=role_index,
                    rows=per_role[role],
                    protocol=protocol,
                    topology=topology,
                    baseline_summary=baseline_map[candidate["model_id"]]["roles"][role],
                    target_modules=target_modules,
                    location=location,
                    evidence_identity=evidence_identity,
                )
                inspection = resume.inspect_role(
                    location,
                    model_slug=model_slug,
                    candidate=candidate,
                    role=role,
                    protocol=protocol,
                    topology=topology,
                    baseline_summary=baseline_map[candidate["model_id"]]["roles"][role],
                    identity=evidence_identity,
                    score_response=topology_runner.score_response,
                    summarize_role=lambda samples, selected_role=role: _summarize_role(
                        topology,
                        candidate,
                        selected_role,
                        samples,
                    ),
                )
                if inspection["state"] != "complete":
                    raise RuntimeError(
                        f"newly produced role evidence failed validation: {role}: "
                        f"{inspection['cross_dose_errors']} "
                        f"{[dose['errors'] for dose in inspection['doses']]}"
                    )
                role_results[role] = _role_payload(inspection)
                comparison = (
                    _reconstruction_comparison(prior_dose_one, inspection["doses"][0])
                    if action == "reconstruct_from_base"
                    else None
                )
                model_audit["roles"][role] = {
                    "action": "recomputed_evidence" if action == "reconstruct_from_base" else "computed_missing_evidence",
                    "reason": action_record["reason"],
                    "prior_standard": _compact_inspection(selection["standard"]),
                    "produced": _compact_inspection(inspection),
                    "reconstruction_comparison": comparison,
                }

            model_result = _build_model_result(candidate, protocol, role_results, runtime)
            results.append(model_result)
            _write_once_json(attempt_root / "model_results" / f"{model_slug}.json", model_result)
            standard_model_result = proof_root / "model_results" / f"{model_slug}.json"
            if not standard_model_result.exists():
                _write_once_json(standard_model_result, model_result)
            print("ELASTICITY_MODEL_COMPLETE_JSON " + json.dumps({"model_id": candidate["model_id"], "mean_final_delta_quality_points": model_result["mean_final_delta_quality_points"], "mean_final_adapted_quality_100": model_result["mean_final_adapted_quality_100"], "mean_delta_per_gpu_hour": model_result["mean_delta_per_gpu_hour"]}, sort_keys=True), flush=True)
        except Exception as exc:
            failure = {"model_id": candidate["model_id"], "revision": candidate["revision"], "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
            failures.append(failure)
            _write_once_json(attempt_root / "runtime" / f"{model_slug}-failure.json", failure)
            print("ELASTICITY_MODEL_FAILURE_JSON " + json.dumps({k: failure[k] for k in ("model_id", "error_type", "error")}, sort_keys=True), flush=True)
        finally:
            del base_model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    _write_once_json(attempt_root / "resume_audit.json", audit)
    if failures or len(results) != len(topology["candidates"]):
        final = {"result_version": "hephaestus-adaptation-elasticity.v1", "status": "failed", "disposition": "incomplete_evidence", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha, "topology_protocol_sha256": topology_sha, "completed_models": len(results), "expected_models": len(topology["candidates"]), "model_results": results, "failures": failures, "resume_audit_ref": str(attempt_root / "resume_audit.json"), "promotion_performed": False, "lineage_mutated": False}
        _write_once_json(attempt_root / "elasticity_result.json", final)
        _write_once_json(execution_root / "driver_result.json", final)
        return 2

    role_rankings: dict[str, Any] = {}
    for role in protocol["roles"]:
        rows = []
        for model in results:
            final_dose = model["roles"][role]["doses"][-1]
            first_dose = model["roles"][role]["doses"][0]
            rows.append({"model_id": model["model_id"], "baseline_quality_100": final_dose["baseline_quality_100"], "dose1_quality_100": first_dose["adapted_quality_100"], "dose2_quality_100": final_dose["adapted_quality_100"], "dose1_delta": first_dose["delta_quality_points"], "dose2_delta": final_dose["delta_quality_points"], "dose2_delta_per_gpu_hour": final_dose["delta_per_gpu_hour"], "dose2_delta_per_million_params": final_dose["delta_per_million_trainable_parameters"], "dose2_delta_per_million_tokens": final_dose["delta_per_million_training_tokens"]})
        role_rankings[role] = {"by_final_quality": sorted(rows, key=lambda row: row["dose2_quality_100"], reverse=True), "by_delta": sorted(rows, key=lambda row: row["dose2_delta"], reverse=True), "by_gpu_hour_elasticity": sorted(rows, key=lambda row: row["dose2_delta_per_gpu_hour"], reverse=True), "by_parameter_elasticity": sorted(rows, key=lambda row: row["dose2_delta_per_million_params"], reverse=True), "by_token_elasticity": sorted(rows, key=lambda row: row["dose2_delta_per_million_tokens"], reverse=True)}

    final = {"result_version": "hephaestus-adaptation-elasticity.v1", "status": "completed", "disposition": "scientific_elasticity_complete", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha, "topology_protocol_sha256": topology_sha, "dataset_sha256": evidence_identity["dataset_sha256"], "contamination_status": "passed", "baseline_run_id": protocol["baseline"]["run_id"], "model_results": results, "role_rankings": role_rankings, "resume_audit_ref": str(attempt_root / "resume_audit.json"), "training_performed": True, "promotion_performed": False, "lineage_mutated": False}
    _write_once_json(attempt_root / "elasticity_result.json", final)
    _preserve_or_initialize_json(proof_root / "elasticity_result.json", final, label="completed elasticity result")
    _write_once_json(execution_root / "driver_result.json", final)
    print("ADAPTATION_ELASTICITY_RESULT_JSON " + json.dumps({"status": final["status"], "disposition": final["disposition"], "role_rankings": role_rankings}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
