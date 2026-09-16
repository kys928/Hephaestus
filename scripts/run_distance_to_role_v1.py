#!/usr/bin/env python3
"""Run one frozen Hephaestus distance-to-role V1 model shard on a mounted Network Volume."""
from __future__ import annotations

import gc
import hashlib
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
PROTOCOL_PATH = ROOT / "configs/experiments/hephaestus_distance_to_role_v1.json"
TOPOLOGY_PATH = ROOT / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"
SEMANTIC_PATH = ROOT / "configs/eval_packs/semantic_behavior_v1.yaml"
ELASTICITY_PROTOCOL_PATH = ROOT / "configs/experiments/hephaestus_adaptation_elasticity_v1.json"
SCIENTIFIC_ROOT = Path("/workspace/hephaestus/scientific/v1")
EPHEMERAL_ROOT = Path("/opt/hephaestus-distance-to-role")


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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


def validate_protocol(protocol: dict[str, Any], topology: dict[str, Any], semantic: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "hephaestus_distance_to_role_v1":
        raise ValueError("unexpected distance-to-role protocol id")
    if protocol.get("scientific_variable") != "cumulative_optimizer_step_dose_under_fixed_role_lora_trajectory":
        raise ValueError("distance-to-role scientific variable drifted")
    if protocol.get("roles") != topology.get("roles"):
        raise ValueError("distance-to-role roles drifted from frozen topology")
    if protocol["sources"]["topology_protocol_sha256"] != sha(TOPOLOGY_PATH.read_bytes()):
        raise ValueError("frozen topology protocol bytes drifted")
    if semantic.get("content_hash") != protocol["sources"]["semantic_content_hash"]:
        raise ValueError("frozen semantic pack identity drifted")
    train = protocol["training"]
    if train.get("dose_optimizer_steps") != [3, 6, 12, 24, 36, 48]:
        raise ValueError("distance dose grid drifted")
    if int(train.get("expected_optimizer_steps_per_epoch", 0)) != 12 or int(train.get("max_epochs", 0)) != 4:
        raise ValueError("distance optimizer-step geometry drifted")
    if int(train["examples_per_role"]) != 48 or int(train["gradient_accumulation_steps"]) != 4:
        raise ValueError("distance role data/batch geometry drifted")
    gov = protocol["governance"]
    if not gov.get("training_is_experimentally_approved") or gov.get("promotion_allowed") or gov.get("lineage_mutation_allowed"):
        raise ValueError("distance governance boundary is invalid")
    shards = protocol["execution"]["model_shards"]
    if set(shards) != {"olmo-granite", "qwen-ministral"}:
        raise ValueError("distance execution shard identities drifted")
    expected = [row["model_id"] for row in topology["candidates"]]
    observed = [*shards["olmo-granite"], *shards["qwen-ministral"]]
    if observed != expected or len(set(observed)) != len(expected):
        raise ValueError("distance shards do not exactly partition frozen candidates")


def load_protocols() -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    protocol_raw = PROTOCOL_PATH.read_bytes()
    topology_raw = TOPOLOGY_PATH.read_bytes()
    protocol = json.loads(protocol_raw)
    topology = json.loads(topology_raw)
    semantic = json.loads(SEMANTIC_PATH.read_text(encoding="utf-8"))
    elasticity = json.loads(ELASTICITY_PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_protocol(protocol, topology, semantic)
    return protocol, sha(protocol_raw), topology, sha(topology_raw), semantic, elasticity


def paths() -> tuple[str, str, int, Path, Path]:
    run_id = required("HEPHAESTUS_DTR_RUN_ID")
    shard = required("HEPHAESTUS_DISTANCE_SHARD")
    attempt = int(required("HEPHAESTUS_ATTEMPT"))
    execution_id = required("HEPHAESTUS_PROOF_RUN_ID")
    proof_root = SCIENTIFIC_ROOT / "distance_to_role" / run_id
    execution_root = SCIENTIFIC_ROOT / "executions" / execution_id / f"attempt-{attempt}"
    return run_id, shard, attempt, proof_root, execution_root


def load_admission(repo_sha: str, protocol_sha: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = SCIENTIFIC_ROOT / "distance_to_role" / "model_admission" / repo_sha
    admission = json.loads((root / "admission.json").read_text(encoding="utf-8"))
    if admission.get("repo_sha") != repo_sha or admission.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("distance admission repository/protocol mismatch")
    if admission.get("training_approved") is not True or admission.get("promotion_allowed") is not False or admission.get("lineage_mutation_allowed") is not False:
        raise RuntimeError("distance admission governance is invalid")
    dataset_raw = (root / "training.jsonl").read_bytes()
    if sha(dataset_raw) != admission.get("training_dataset_sha256"):
        raise RuntimeError("distance admission dataset hash mismatch")
    model_admission = json.loads((root / "source_model_admission.json").read_text(encoding="utf-8"))
    if model_admission.get("candidate_order") != admission.get("candidate_order"):
        raise RuntimeError("distance source model admission candidate mismatch")
    rows = [json.loads(line) for line in dataset_raw.decode("utf-8").splitlines() if line.strip()]
    return admission, model_admission, rows


def semantic_tasks(semantic: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for key in ("generation_probes", "continuation_prompts", "structure_tests", "repetition_checks", "length_termination_checks"):
        for task in semantic.get(key, []):
            tasks.append(dict(task))
    ids = [str(row.get("task_id")) for row in tasks]
    if ids != ["instruction_triplet", "planet_fact", "observatory_continuation", "structured_planet_answer", "anti_repetition", "brief_termination"]:
        raise RuntimeError(f"semantic task identity/order drifted: {ids}")
    return tasks


def evaluate_role(model: Any, tokenizer: Any, topology: dict[str, Any], candidate: dict[str, Any], role: str, seeds: list[int], sample_path: Path | None) -> dict[str, Any]:
    role_cases = [case for case in topology["cases"] if case["role"] == role]
    role_spec = dict(topology)
    role_spec["roles"] = [role]
    role_spec["generation"] = dict(topology["generation"])
    role_spec["generation"]["seeds"] = list(seeds)
    samples: list[dict[str, Any]] = []
    tokenizer.padding_side = "left"
    model.eval()
    for seed in seeds:
        for case in role_cases:
            prompt = topology_runner._prompt(topology, case)
            generated = topology_runner._generate_one(model, tokenizer, prompt, seed=int(seed), max_new_tokens=int(topology["generation"]["max_new_tokens"]))
            score = topology_runner.score_response(topology, case, generated["output"])
            row = {
                "model_id": candidate["model_id"], "revision": candidate["revision"], "case_id": case["case_id"],
                "role": role, "condition": case["condition"], "pair_id": case.get("pair_id"), "seed": int(seed),
                "output": generated["output"], "generation": {k: v for k, v in generated.items() if k != "output"}, "score": score
            }
            samples.append(row)
            if sample_path is not None:
                append_jsonl(sample_path, row)
    expected = len(role_cases) * len(seeds)
    if len(samples) != expected:
        raise RuntimeError(f"incomplete role screen: {len(samples)} != {expected}")
    summary = topology_runner._summarize_model(role_spec, candidate, samples, {})["roles"][role]
    tokenizer.padding_side = "right"
    return {"summary": summary, "samples": samples}


def evaluate_semantic(model: Any, tokenizer: Any, semantic: dict[str, Any], candidate: dict[str, Any], seeds: list[int], sample_path: Path | None) -> dict[str, Any]:
    tasks = semantic_tasks(semantic)
    rows: list[dict[str, Any]] = []
    hard_failures: set[str] = set()
    tokenizer.padding_side = "left"
    model.eval()
    for seed in seeds:
        for task in tasks:
            generated = topology_runner._generate_one(model, tokenizer, str(task["prompt"]), seed=int(seed), max_new_tokens=int(semantic["decoding_config"]["max_new_tokens"]))
            score = evaluate_behavioral_sample(task, generated["output"], seed)
            for check in score.checks:
                if check.hard and not check.passed:
                    hard_failures.add(f"{task['task_id']}:{seed}:{check.name}")
            row = {
                "model_id": candidate["model_id"], "revision": candidate["revision"], "task_id": task["task_id"], "seed": int(seed),
                "output": generated["output"], "generation": {k: v for k, v in generated.items() if k != "output"}, "score": score.to_dict()
            }
            rows.append(row)
            if sample_path is not None:
                append_jsonl(sample_path, row)
    expected = len(tasks) * len(seeds)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete semantic evidence: {len(rows)} != {expected}")
    summary = {
        "sample_count": len(rows), "task_count": len(tasks),
        "mean_score": mean([float(row["score"]["overall_score"]) for row in rows]),
        "deterministic_pass_rate": mean([float(row["score"]["deterministic_passed"]) for row in rows]),
        "hard_failures": sorted(hard_failures), "hard_failure_count": len(hard_failures)
    }
    tokenizer.padding_side = "right"
    return {"summary": summary, "samples": rows}


def safe_state(role_summary: dict[str, Any], semantic_summary: dict[str, Any], baseline_semantic: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    base_failures = set(baseline_semantic["hard_failures"])
    adapted_failures = set(semantic_summary["hard_failures"])
    new_hard = sorted(adapted_failures - base_failures)
    semantic_delta = float(semantic_summary["mean_score"]) - float(baseline_semantic["mean_score"])
    gates = protocol["role_screening"]["threshold_hard_gates"]
    interface_ok = float(role_summary["schema_compliance"]) >= float(gates["schema_compliance"]) and float(role_summary["hallucination_rate"]) <= float(gates["hallucination_rate"])
    strict_cfg = protocol["semantic_regression"]["strict_non_regression"]
    bounded_cfg = protocol["semantic_regression"]["bounded_non_regression"]
    strict_safe = interface_ok and not new_hard and semantic_delta >= float(strict_cfg["minimum_score_delta"]) - 1e-12
    bounded_safe = interface_ok and not new_hard and semantic_delta >= float(bounded_cfg["minimum_score_delta"]) - 1e-12
    return {"interface_ok": interface_ok, "new_hard_failures": new_hard, "semantic_score_delta": semantic_delta, "strict_safe": strict_safe, "bounded_safe": bounded_safe}


def threshold_distances(points: list[dict[str, Any]], thresholds: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"raw": {}, "strict_safe": {}, "bounded_safe": {}}
    for threshold in thresholds:
        key = str(threshold)
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
                    "optimizer_steps": int(point["optimizer_steps"]), "approx_epochs": int(point["optimizer_steps"]) / 12.0,
                    "supervised_tokens": int(point["training"]["supervised_tokens_cumulative"]),
                    "quality_100": float(point["role"]["quality_100"]), "semantic_score_delta": float(point["safety"]["semantic_score_delta"])
                }
                break
            out[mode][key] = match
    return out


def select_points(points: list[dict[str, Any]]) -> dict[str, int | None]:
    def best(rows: list[dict[str, Any]]) -> int | None:
        if not rows:
            return None
        winner = max(rows, key=lambda row: (float(row["role"]["quality_100"]), -int(row["optimizer_steps"])))
        return int(winner["optimizer_steps"])
    return {
        "raw_peak_steps": best([row for row in points if row["safety"]["interface_ok"]]),
        "strict_safe_peak_steps": best([row for row in points if row["safety"]["strict_safe"]]),
        "bounded_safe_peak_steps": best([row for row in points if row["safety"]["bounded_safe"]])
    }


def validate_existing_role_curve(path: Path, *, protocol_sha: str, dataset_sha: str, candidate: dict[str, Any], role: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {"protocol_sha256": protocol_sha, "training_dataset_sha256": dataset_sha, "model_id": candidate["model_id"], "revision": candidate["revision"], "role": role, "status": "complete"}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"persisted role curve mismatch {path}: {key}")
    if [int(row["optimizer_steps"]) for row in payload.get("points", [])] != [0, 3, 6, 12, 24, 36, 48]:
        raise RuntimeError(f"persisted role curve dose grid is incomplete: {path}")
    evidence_files = payload.get("evidence_files")
    if not isinstance(evidence_files, dict) or not evidence_files:
        raise RuntimeError(f"persisted role curve lacks evidence component hashes: {path}")
    root = path.parent
    for relative, expected_hash in evidence_files.items():
        evidence_path = root / str(relative)
        if not evidence_path.is_file() or "sha256:" + sha_file(evidence_path) != str(expected_hash):
            raise RuntimeError(f"persisted role evidence component failed verification: {evidence_path}")
    return payload


def adapter_manifest(adapter_dir: Path, **metadata: Any) -> dict[str, Any]:
    components: dict[str, str] = {}
    total = 0
    for path in sorted(adapter_dir.rglob("*")):
        if path.is_file() and path.name != "hephaestus_distance_adapter_manifest.json":
            components[path.relative_to(adapter_dir).as_posix()] = "sha256:" + sha_file(path)
            total += path.stat().st_size
    return {
        "manifest_version": "distance-to-role-adapter.v1", "components": components, "adapter_bytes": total,
        "manifest_sha256": "sha256:" + sha(json.dumps(components, sort_keys=True, separators=(",", ":")).encode()), **metadata
    }


def copy_snapshot_manifest(ephemeral_manifest: dict[str, Any], proof_root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in ephemeral_manifest.items() if key != "snapshot_path"}
    compact["snapshot_path"] = "ephemeral://pod/model-snapshot"
    compact["persistent_model_cache"] = False
    target = proof_root / "model_manifests" / slug(candidate["model_id"]) / "snapshot_manifest.json"
    if target.exists():
        observed = json.loads(target.read_text(encoding="utf-8"))
        if observed != compact:
            raise RuntimeError("persisted model manifest differs from current immutable materialization")
    else:
        write_once_json(target, compact)
    return compact


def train_role_curve(*, base_model: Any, tokenizer: Any, candidate: dict[str, Any], role: str, role_index: int, rows: list[dict[str, Any]], protocol: dict[str, Any], protocol_sha: str, topology: dict[str, Any], semantic: dict[str, Any], baseline_topology_role: dict[str, Any], baseline_semantic: dict[str, Any], target_modules: list[str], work_dir: Path, dataset_sha: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import get_linear_schedule_with_warmup

    train = protocol["training"]
    screening_seeds = [int(v) for v in protocol["role_screening"]["seeds"]]
    semantic_seeds = [int(v) for v in protocol["semantic_regression"]["seeds"]]
    confirmation_seeds = [int(v) for v in protocol["role_screening"]["full_confirmation_seeds"]]
    work_dir.mkdir(parents=True, exist_ok=False)

    dose0_role = evaluate_role(base_model, tokenizer, topology, candidate, role, screening_seeds, work_dir / "samples" / "role_step_000.jsonl")
    observed_base = float(dose0_role["summary"]["quality_100"])
    expected_base = float(baseline_topology_role["quality_100"])
    if not math.isclose(observed_base, expected_base, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"dose-zero topology drift for {candidate['model_id']} role={role}: {observed_base} != {expected_base}")

    zero_safety = safe_state(dose0_role["summary"], baseline_semantic, baseline_semantic, protocol)
    points: list[dict[str, Any]] = [{
        "optimizer_steps": 0, "approx_epochs": 0.0, "role": dose0_role["summary"], "semantic": baseline_semantic, "safety": zero_safety,
        "training": {"training_seconds_cumulative": 0.0, "supervised_tokens_cumulative": 0, "trainable_parameters": 0, "peak_vram_bytes": int(torch.cuda.max_memory_allocated())},
        "adapter": None
    }]

    torch.manual_seed(int(train["model_seed"]) + role_index)
    torch.cuda.manual_seed_all(int(train["model_seed"]) + role_index)
    shuffled = list(rows)
    random.Random(int(train["shuffle_seed"]) + role_index).shuffle(shuffled)
    encoded = [elastic._tokenize_example(tokenizer, row, int(train["max_seq_length"])) for row in shuffled]
    lora_config = LoraConfig(r=int(train["rank"]), lora_alpha=int(train["alpha"]), lora_dropout=float(train["dropout"]), bias=str(train["bias"]), task_type=TaskType.CAUSAL_LM, target_modules=target_modules)
    peft_model = get_peft_model(base_model, lora_config)
    trainable_parameters = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if trainable_parameters <= 0:
        raise RuntimeError("no trainable LoRA parameters")
    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=float(train["learning_rate"]), weight_decay=float(train["weight_decay"]))
    max_steps = max(int(v) for v in train["dose_optimizer_steps"])
    warmup_steps = max(1, int(round(max_steps * float(train["warmup_ratio"]))))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    accumulation = int(train["gradient_accumulation_steps"])
    dose_steps = set(int(v) for v in train["dose_optimizer_steps"])
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
        for idx, item in enumerate(encoded):
            started = time.perf_counter()
            peft_model.train()
            tokenizer.padding_side = "right"
            batch = elastic._tensorize(item)
            cumulative_tokens += sum(1 for label in item["labels"] if label != -100)
            outputs = peft_model(**batch)
            loss = outputs.loss / accumulation
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite distance LoRA loss for {candidate['model_id']} role={role}")
            loss.backward()
            segment_losses.append(float(loss.detach().item()) * accumulation)
            should_update = ((idx + 1) % accumulation == 0)
            if not should_update:
                cumulative_seconds += time.perf_counter() - started
                continue
            torch.nn.utils.clip_grad_norm_([p for p in peft_model.parameters() if p.requires_grad], float(train["max_grad_norm"]))
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            cumulative_seconds += time.perf_counter() - started
            optimizer_step += 1
            if optimizer_step not in dose_steps:
                continue

            adapter_dir = work_dir / "adapters" / f"step_{optimizer_step:03d}"
            peft_model.save_pretrained(adapter_dir, safe_serialization=True)
            manifest = adapter_manifest(
                adapter_dir, protocol_id=protocol["protocol_id"], protocol_sha256=protocol_sha,
                training_dataset_sha256=dataset_sha, model_id=candidate["model_id"], revision=candidate["revision"], role=role,
                optimizer_steps=optimizer_step, trainable_parameters=trainable_parameters,
                target_module_count=len(target_modules), target_modules_sha256=sha("\n".join(target_modules).encode())
            )
            write_once_json(adapter_dir / "hephaestus_distance_adapter_manifest.json", manifest)
            role_eval = evaluate_role(peft_model, tokenizer, topology, candidate, role, screening_seeds, work_dir / "samples" / f"role_step_{optimizer_step:03d}.jsonl")
            sem_eval = evaluate_semantic(peft_model, tokenizer, semantic, candidate, semantic_seeds, work_dir / "samples" / f"semantic_step_{optimizer_step:03d}.jsonl")
            safety = safe_state(role_eval["summary"], sem_eval["summary"], baseline_semantic, protocol)
            points.append({
                "optimizer_steps": optimizer_step, "approx_epochs": optimizer_step / float(train["expected_optimizer_steps_per_epoch"]),
                "role": role_eval["summary"], "semantic": sem_eval["summary"], "safety": safety,
                "training": {"training_seconds_cumulative": cumulative_seconds, "supervised_tokens_cumulative": cumulative_tokens, "trainable_parameters": trainable_parameters, "peak_vram_bytes": int(torch.cuda.max_memory_allocated()), "mean_loss_since_previous_dose": mean(segment_losses)},
                "adapter": manifest
            })
            segment_losses = []
            print("DISTANCE_DOSE_COMPLETE_JSON " + json.dumps({
                "model_id": candidate["model_id"], "role": role, "optimizer_steps": optimizer_step,
                "quality_100": role_eval["summary"]["quality_100"], "semantic_mean_score": sem_eval["summary"]["mean_score"],
                "new_hard_failures": safety["new_hard_failures"], "strict_safe": safety["strict_safe"], "bounded_safe": safety["bounded_safe"]
            }, sort_keys=True), flush=True)
            if optimizer_step >= max_steps:
                stop = True
                break
    if optimizer_step != max_steps:
        raise RuntimeError(f"distance trajectory stopped at {optimizer_step} != {max_steps}")

    selections = select_points(points)
    thresholds = threshold_distances(points, [int(v) for v in protocol["role_screening"]["quality_thresholds_100"]])
    base_model = peft_model.unload()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    gc.collect(); torch.cuda.empty_cache()

    confirmations: dict[str, Any] = {}
    selected_steps = sorted({int(step) for step in selections.values() if step is not None})
    for step in selected_steps:
        if step == 0:
            full = evaluate_role(base_model, tokenizer, topology, candidate, role, confirmation_seeds, work_dir / "confirmations" / "role_step_000.jsonl")
        else:
            adapter_dir = work_dir / "adapters" / f"step_{step:03d}"
            adapted = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
            full = evaluate_role(adapted, tokenizer, topology, candidate, role, confirmation_seeds, work_dir / "confirmations" / f"role_step_{step:03d}.jsonl")
            base_model = adapted.unload()
            for parameter in base_model.parameters():
                parameter.requires_grad_(False)
            gc.collect(); torch.cuda.empty_cache()
        confirmations[str(step)] = full["summary"]

    evidence_files = {
        path.relative_to(work_dir).as_posix(): "sha256:" + sha_file(path)
        for path in sorted(work_dir.rglob("*")) if path.is_file() and path.name != "role_curve.json"
    }
    curve = {
        "result_version": "distance-to-role-role-curve.v1", "status": "complete",
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha, "training_dataset_sha256": dataset_sha,
        "model_id": candidate["model_id"], "revision": candidate["revision"], "role": role,
        "baseline_topology_quality_100": expected_base, "points": points, "threshold_distances": thresholds,
        "selections": selections, "full_seed_confirmations": confirmations, "evidence_files": evidence_files,
        "promotion_performed": False, "lineage_mutated": False
    }
    write_once_json(work_dir / "role_curve.json", curve)
    return base_model, curve


def build_shard_result(*, run_id: str, shard: str, protocol: dict[str, Any], protocol_sha: str, topology_sha: str, admission: dict[str, Any], model_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "result_version": "distance-to-role-shard-result.v1", "status": "completed", "disposition": "scientific_distance_shard_complete",
        "run_id": run_id, "shard": shard, "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha,
        "topology_protocol_sha256": topology_sha, "semantic_content_hash": protocol["sources"]["semantic_content_hash"],
        "training_dataset_sha256": admission["training_dataset_sha256"], "models": model_results,
        "training_performed": True, "promotion_performed": False, "lineage_mutated": False
    }


def main() -> int:
    import torch

    protocol, protocol_sha, topology, topology_sha, semantic, _ = load_protocols()
    run_id, shard, attempt, proof_root, execution_root = paths()
    repo_sha = required("HEPHAESTUS_REPO_SHA")
    if shard not in protocol["execution"]["model_shards"]:
        raise RuntimeError(f"unknown distance shard: {shard}")
    proof_root.mkdir(parents=True, exist_ok=True)
    execution_root.mkdir(parents=True, exist_ok=True)
    terminal = execution_root / "driver_result.json"
    if terminal.exists():
        raise RuntimeError("distance execution attempt already has terminal result")

    result: dict[str, Any] = {
        "result_version": "distance-to-role-driver.v1", "status": "running", "run_id": run_id, "shard": shard,
        "attempt": attempt, "repo_sha": repo_sha, "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha,
        "training_performed": False, "promotion_performed": False, "lineage_mutated": False
    }
    try:
        admission, model_admission, dataset = load_admission(repo_sha, protocol_sha)
        if admission.get("topology_protocol_sha256") != topology_sha:
            raise RuntimeError("distance admission topology hash mismatch")
        dataset_sha = admission["training_dataset_sha256"]
        per_role = {role: [row for row in dataset if row.get("role") == role] for role in protocol["roles"]}
        expected_count = int(protocol["training"]["examples_per_role"])
        if any(len(rows) != expected_count for rows in per_role.values()):
            raise RuntimeError("distance training dataset role counts drifted")

        baseline = elastic._load_baseline({"baseline": {"run_id": protocol["sources"]["topology_run_id"]}}, topology_sha)
        baseline_map = {row["model_id"]: row for row in baseline["model_summaries"]}
        selected_ids = list(protocol["execution"]["model_shards"][shard])
        candidates = [row for row in topology["candidates"] if row["model_id"] in selected_ids]
        if [row["model_id"] for row in candidates] != selected_ids:
            raise RuntimeError("distance shard candidate order/identity mismatch")

        model_results: list[dict[str, Any]] = []
        for candidate in candidates:
            model_slug = slug(candidate["model_id"])
            model_root = proof_root / "models" / model_slug
            role_results: dict[str, Any] = {}
            missing_roles = []
            for role in protocol["roles"]:
                curve_path = model_root / "roles" / role / "role_curve.json"
                existing = validate_existing_role_curve(curve_path, protocol_sha=protocol_sha, dataset_sha=dataset_sha, candidate=candidate, role=role)
                if existing is None:
                    missing_roles.append(role)
                else:
                    role_results[role] = existing
            if not missing_roles:
                model_results.append({"model_id": candidate["model_id"], "revision": candidate["revision"], "status": "complete", "roles": role_results, "evidence_reused": True})
                print("DISTANCE_MODEL_REUSED_JSON " + json.dumps({"model_id": candidate["model_id"], "roles": list(role_results)}, sort_keys=True), flush=True)
                continue

            ephemeral_model_root = EPHEMERAL_ROOT / run_id / shard / f"attempt-{attempt}" / model_slug
            if ephemeral_model_root.exists():
                shutil.rmtree(ephemeral_model_root)
            ephemeral_model_root.mkdir(parents=True, exist_ok=False)
            snapshot, ephemeral_manifest = topology_runner._materialize(candidate, model_admission, ephemeral_model_root)
            compact_manifest = copy_snapshot_manifest(ephemeral_manifest, proof_root, candidate)
            base_model, tokenizer, runtime = elastic._load_training_base(snapshot, candidate, {"training": protocol["training"]})
            topology_runner._warmup(base_model, tokenizer)
            runtime.update({"model_id": candidate["model_id"], "revision": candidate["revision"], "snapshot_manifest_hash": compact_manifest["manifest_hash"], "shard": shard, "attempt": attempt})
            runtime_path = execution_root / "runtime" / f"{model_slug}.json"
            write_once_json(runtime_path, runtime)
            target_modules = elastic._target_modules(base_model, list(protocol["training"]["target_module_suffixes"]))

            base_sem_path = model_root / "base_semantic.json"
            base_samples_path = model_root / "base_semantic_samples.jsonl"
            if base_sem_path.exists():
                base_semantic_record = json.loads(base_sem_path.read_text(encoding="utf-8"))
                if base_semantic_record.get("protocol_sha256") != protocol_sha or base_semantic_record.get("revision") != candidate["revision"]:
                    raise RuntimeError("persisted base semantic evidence drifted")
                if not base_samples_path.is_file() or base_semantic_record.get("samples_sha256") != "sha256:" + sha_file(base_samples_path):
                    raise RuntimeError("persisted base semantic samples failed hash verification")
                baseline_semantic = base_semantic_record["summary"]
            else:
                temp_samples = execution_root / "base_semantic" / f"{model_slug}.jsonl"
                base_eval = evaluate_semantic(base_model, tokenizer, semantic, candidate, [int(v) for v in protocol["semantic_regression"]["seeds"]], temp_samples)
                baseline_semantic = base_eval["summary"]
                if base_samples_path.exists():
                    orphan = proof_root / "orphaned_partial_evidence" / f"{model_slug}-base-semantic-before-attempt-{attempt}.jsonl"
                    orphan.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(base_samples_path, orphan)
                base_samples_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp_samples, base_samples_path)
                write_once_json(base_sem_path, {
                    "protocol_sha256": protocol_sha, "semantic_content_hash": protocol["sources"]["semantic_content_hash"],
                    "model_id": candidate["model_id"], "revision": candidate["revision"],
                    "samples_sha256": "sha256:" + sha_file(base_samples_path), "summary": baseline_semantic
                })

            for role_index, role in enumerate(protocol["roles"]):
                if role in role_results:
                    continue
                canonical_dir = model_root / "roles" / role
                work_dir = execution_root / "work" / model_slug / role
                if work_dir.exists():
                    raise RuntimeError(f"attempt work directory already exists: {work_dir}")
                base_model, curve = train_role_curve(
                    base_model=base_model, tokenizer=tokenizer, candidate=candidate, role=role, role_index=role_index,
                    rows=per_role[role], protocol=protocol, protocol_sha=protocol_sha, topology=topology, semantic=semantic,
                    baseline_topology_role=baseline_map[candidate["model_id"]]["roles"][role], baseline_semantic=baseline_semantic,
                    target_modules=target_modules, work_dir=work_dir, dataset_sha=dataset_sha
                )
                canonical_dir.parent.mkdir(parents=True, exist_ok=True)
                if canonical_dir.exists():
                    raise RuntimeError(f"canonical role evidence appeared concurrently: {canonical_dir}")
                os.replace(work_dir, canonical_dir)
                role_results[role] = curve
                print("DISTANCE_ROLE_COMPLETE_JSON " + json.dumps({"model_id": candidate["model_id"], "role": role, "selections": curve["selections"], "threshold_distances": curve["threshold_distances"]}, sort_keys=True), flush=True)

            model_results.append({"model_id": candidate["model_id"], "revision": candidate["revision"], "status": "complete", "roles": role_results, "evidence_reused": False})
            del base_model, tokenizer
            gc.collect(); torch.cuda.empty_cache()
            shutil.rmtree(ephemeral_model_root, ignore_errors=True)

        shard_result = build_shard_result(run_id=run_id, shard=shard, protocol=protocol, protocol_sha=protocol_sha, topology_sha=topology_sha, admission=admission, model_results=model_results)
        shard_path = proof_root / "shards" / shard / "shard_result.json"
        if shard_path.exists():
            observed = json.loads(shard_path.read_text(encoding="utf-8"))
            if observed != shard_result:
                raise RuntimeError("persisted shard result differs from recomputed complete shard")
        else:
            write_once_json(shard_path, shard_result)
        result.update(shard_result)
        result["status"] = "completed"
        result["training_performed"] = True
        print("DISTANCE_SHARD_COMPLETE_JSON " + json.dumps({"run_id": run_id, "shard": shard, "model_ids": [row["model_id"] for row in model_results]}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["disposition"] = "runtime_or_evidence_failure"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        atomic_json(terminal, result)


if __name__ == "__main__":
    raise SystemExit(main())
