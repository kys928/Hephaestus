#!/usr/bin/env python3
"""Benchmark the frozen V2 Evaluator on the exact V3.1 neutral holdout."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import run_interface_repair_v1 as v1
import run_interface_repair_v2 as v2
import run_interface_repair_v3_evaluator as v3
import run_interface_repair_v3_1_evaluator as v31
from interface_repair_v2_bootstrap import load_stack_compatible

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_repair_v2_neutral_benchmark.json"
ROLE = "evaluator"
PARTITION = "neutral_holdout"


def load_cfg() -> dict[str, Any]:
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    if cfg.get("diagnostic_only") is not True or cfg.get("training_performed") is not False:
        raise RuntimeError("V2 neutral benchmark must remain diagnostic-only")
    g = cfg["governance"]
    if g.get("training_allowed") is not False or g.get("adapter_mutation_allowed") is not False:
        raise RuntimeError("V2 neutral benchmark may not train or mutate adapters")
    if g.get("production_promotion_allowed") or g.get("automatic_role_dispatch_allowed") or g.get("certification_claim_allowed"):
        raise RuntimeError("V2 neutral benchmark may not promote, dispatch, or certify")
    return cfg


def load_pack(cfg: Mapping[str, Any]) -> dict[str, Any]:
    path = ROOT / str(cfg["pack"]["builder_path"])
    spec = importlib.util.spec_from_file_location("neutral_holdout_pack", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import neutral holdout builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pack = module.build_pack()
    module.validate(pack)
    observed = module.canonical_sha256(pack)
    if observed != str(cfg["pack"]["canonical_sha256"]):
        raise RuntimeError(f"neutral pack hash drift: {observed}")
    return pack


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_stack(cfg: Mapping[str, Any]) -> dict[str, Any]:
    stack = load_stack_compatible(cfg)
    source = dict(cfg["source_v2_adapter"])
    evaluator = dict(stack["roles"][ROLE])
    if evaluator.get("model_id") != source.get("model_id") or evaluator.get("revision") != source.get("revision"):
        raise RuntimeError("V2 source model/revision mismatch")
    evaluator["adapter"] = {"s3_key": source["s3_key"], "sha256": source["sha256"]}
    evaluator["status"] = "frozen_v2_neutral_parent_benchmark"
    roles = dict(stack["roles"])
    roles[ROLE] = evaluator
    out = dict(stack)
    out["roles"] = roles
    return out


def materialize_frozen_v2(client: Any, role_spec: Mapping[str, Any], root: Path) -> tuple[Path, Path, int]:
    from huggingface_hub import snapshot_download

    model_id = str(role_spec["model_id"])
    revision = str(role_spec["revision"])
    try:
        snapshot = snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
    except Exception:
        cache = Path("/opt/hephaestus-hf-cache")
        cache.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_download(repo_id=model_id, revision=revision, cache_dir=str(cache))
    base = Path(snapshot)
    if not (base / "config.json").is_file():
        raise RuntimeError("Granite snapshot incomplete")

    adapter = role_spec["adapter"]
    expected_sha = str(adapter["sha256"])
    cache_root = Path(os.environ.get("HEPHAESTUS_V2_ADAPTER_CACHE", "/workspace/hephaestus-cache/interface-repair-v2-neutral/adapters"))
    entry = cache_root / expected_sha
    entry.mkdir(parents=True, exist_ok=True)
    archive = entry / "parent-adapter.tar.gz"
    if not archive.is_file() or _sha256(archive) != expected_sha:
        tmp = entry / f"download-{os.getpid()}-{time.time_ns()}.tmp"
        try:
            client.download_file(v1.bucket(), str(adapter["s3_key"]), str(tmp))
            if _sha256(tmp) != expected_sha:
                raise RuntimeError("V2 adapter SHA verification failed")
            os.replace(tmp, archive)
        finally:
            tmp.unlink(missing_ok=True)
    observed_bytes = archive.stat().st_size
    extracted = entry / "extracted"
    adapter_dir = extracted / "adapter"
    ready = entry / ".ready"
    if not (ready.is_file() and adapter_dir.is_dir() and ready.read_text().strip() == expected_sha):
        staging = entry / f"extract-{os.getpid()}-{time.time_ns()}"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            observed = v1.safe_extract(archive, staging)
            if observed.name != "adapter" or observed.parent != staging:
                normalized = staging / "adapter"
                if normalized.exists():
                    shutil.rmtree(normalized)
                shutil.move(str(observed), str(normalized))
            if extracted.exists():
                shutil.rmtree(extracted)
            os.replace(staging, extracted)
            ready.write_text(expected_sha + "\n", encoding="utf-8")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    if not adapter_dir.is_dir():
        raise RuntimeError("V2 adapter extraction incomplete")
    return base, adapter_dir, observed_bytes


def _model_load_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(cfg))
    clone["training"] = {
        "lora_rank": int(cfg["model_protocol"]["lora_rank"]),
        "lora_alpha": int(cfg["model_protocol"]["lora_alpha"]),
    }
    return clone


def _eligible(summary: Mapping[str, float], cfg: Mapping[str, Any]) -> bool:
    e = cfg["parent_selection"]["eligibility"]
    return (
        float(summary["schema_compliance"]) >= float(e["schema_compliance_required"])
        and float(summary["evidence_grounding"]) >= float(e["minimum_evidence_grounding"])
        and float(summary["hallucination_rate"]) <= float(e["maximum_hallucination_rate"])
        and float(summary["boundary_projection_rate"]) >= float(e["boundary_projection_rate_required"])
        and float(summary["system_semantic_escalation_rate"]) <= float(e["maximum_system_semantic_escalation_rate"])
    )


def _rank(m: Mapping[str, float]) -> tuple[float, ...]:
    return (
        float(m["decision_exact_pass_rate"]),
        float(m["semantic_exact_pass_rate"]),
        float(m["semantic_quality_100"]) / 100.0,
        -float(m["hallucination_rate"]),
        float(m["system_action_exact_pass_rate"]),
    )


def analyze(rows: list[dict[str, Any]], summary: Mapping[str, float], cfg: Mapping[str, Any]) -> dict[str, Any]:
    confusion: Counter[str] = Counter()
    wrong = 0
    wrong_regressed = 0
    for row in rows:
        expected = str(row["expected"]["decision"])
        parsed = row["score"].get("parsed")
        predicted = str(parsed.get("decision", "<invalid>")) if isinstance(parsed, dict) else "<invalid>"
        confusion[f"{expected} -> {predicted}"] += 1
        if predicted != expected:
            wrong += 1
            if predicted == "regressed":
                wrong_regressed += 1

    ref = dict(cfg["reference_v31_neutral"])
    v2_eligible = _eligible(summary, cfg)
    v31_eligible = _eligible(ref, cfg)
    if v2_eligible and not v31_eligible:
        winner = "v2"
    elif v31_eligible and not v2_eligible:
        winner = "v3_1"
    elif v2_eligible and v31_eligible:
        rv2, rv31 = _rank(summary), _rank(ref)
        winner = "v2" if rv2 >= rv31 else "v3_1"
    else:
        rv2, rv31 = _rank(summary), _rank(ref)
        winner = "v2" if rv2 >= rv31 else "v3_1"

    return {
        "decision_confusion": dict(sorted(confusion.items())),
        "wrong_decision_count": wrong,
        "wrong_decisions_to_regressed": wrong_regressed,
        "v2_eligible_parent": v2_eligible,
        "v31_eligible_parent": v31_eligible,
        "selected_parent": winner,
        "ranking_order": list(cfg["parent_selection"]["ranking_order"]),
        "reference_v31_neutral": ref,
        "delta_v2_minus_v31": {
            "decision_exact_pass_rate": float(summary["decision_exact_pass_rate"]) - float(ref["decision_exact_pass_rate"]),
            "semantic_exact_pass_rate": float(summary["semantic_exact_pass_rate"]) - float(ref["semantic_exact_pass_rate"]),
            "semantic_quality_100": float(summary["semantic_quality_100"]) - float(ref["semantic_quality_100"]),
            "hallucination_rate": float(summary["hallucination_rate"]) - float(ref["hallucination_rate"]),
            "system_action_exact_pass_rate": float(summary["system_action_exact_pass_rate"]) - float(ref["system_action_exact_pass_rate"]),
        },
    }


def main() -> int:
    cfg = load_cfg()
    run_id = (os.environ.get("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID") or "").strip()
    if not run_id:
        raise RuntimeError("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID is required")
    prefix = f"{str(cfg['execution']['s3_prefix']).rstrip('/')}/{run_id}"
    progress_key = f"{prefix}/progress.json"
    result_key = f"{prefix}/result.json"
    client = v1.s3_client()
    client.head_bucket(Bucket=v1.bucket())
    deadline = time.monotonic() + float(cfg["execution"]["hard_wall_seconds"])

    def heartbeat(stage: str, **extra: Any) -> None:
        payload = {"run_id": run_id, "stage": stage, "timestamp_unix": time.time(), "remaining_wall_seconds": max(0.0, deadline-time.monotonic()), **extra}
        v2.put_json(client, progress_key, payload)
        print("INTERFACE_REPAIR_V2_NEUTRAL_PROGRESS_JSON " + json.dumps(payload, sort_keys=True), flush=True)

    try:
        pack = load_pack(cfg)
        cases = list(pack["partitions"][ROLE][PARTITION])
        if len(cases) != int(cfg["pack"]["cases"]):
            raise RuntimeError("neutral holdout count drift")
        stack = load_frozen_stack(cfg)
        role_spec = stack["roles"][ROLE]
        root = Path("/opt/hephaestus-v2-neutral-benchmark")
        root.mkdir(parents=True, exist_ok=True)
        heartbeat("materializing_frozen_v2_adapter", adapter_sha256=role_spec["adapter"]["sha256"])
        base_dir, adapter_dir, adapter_bytes = materialize_frozen_v2(client, role_spec, root)
        model, tokenizer, runtime_info = v1.load_model(ROLE, _model_load_cfg(cfg), role_spec, base_dir, adapter_dir)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable != 0:
            raise RuntimeError("benchmark unexpectedly exposes trainable parameters")

        runtime = dict(cfg["roles"][ROLE])
        rows: list[dict[str, Any]] = []
        shard: list[dict[str, Any]] = []
        shard_index = 0
        seed = int(cfg["model_protocol"]["seed"])
        shard_size = int(cfg["execution"]["evaluation_shard_size"])
        heartbeat("neutral_evaluation_started", cases=len(cases), trainable_parameters=0)
        for i, case in enumerate(cases, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("V2 neutral benchmark hard wall reached")
            text = v1.prompt(pack, case, ROLE)
            generation = v31.generate_v31(model, tokenizer, runtime, text, seed+i, deadline)
            score = v3.score_case_v3(pack, case, ROLE, str(generation["projected_output"]))
            row = {"case_id": case["case_id"], "source_case_id": case["source_case_id"], "role": ROLE, "phase": PARTITION, "kind": case.get("kind"), "context": case.get("context"), "semantic_root": case.get("semantic_root"), "generation": generation, "score": score, "expected": case["expected"]}
            rows.append(row)
            shard.append(row)
            if len(shard) >= shard_size or i == len(cases):
                shard_index += 1
                v2.put_jsonl(client, f"{prefix}/samples/shard-{shard_index:04d}.jsonl", shard)
                shard = []
            if i == 1 or i % 32 == 0 or i == len(cases):
                partial = v31.summarize_v31(rows)
                heartbeat("neutral_evaluation_progress", completed=i, total=len(cases), semantic_quality_100=partial["semantic_quality_100"], decision_exact_pass_rate=partial["decision_exact_pass_rate"])

        summary = v31.summarize_v31(rows)
        analysis = analyze(rows, summary, cfg)
        result = {
            "result_version": "hephaestus-interface-repair-v2-neutral-benchmark.v1",
            "status": "completed",
            "run_id": run_id,
            "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA"),
            "pack_sha256": cfg["pack"]["canonical_sha256"],
            "source_v2_adapter": {**dict(cfg["source_v2_adapter"]), "bytes": adapter_bytes},
            "training_performed": False,
            "adapter_mutated": False,
            "adapter_persisted": False,
            "certification_claim_performed": False,
            "production_promotion_performed": False,
            "automatic_role_dispatch_performed": False,
            "runtime": {**runtime_info, "trainable_parameter_count_after_freeze": trainable},
            "summary": summary,
            "analysis": analysis,
            "completed_at_unix": time.time()
        }
        v2.put_json(client, result_key, result)
        heartbeat("complete", selected_parent=analysis["selected_parent"], semantic_quality_100=summary["semantic_quality_100"], decision_exact_pass_rate=summary["decision_exact_pass_rate"], wrong_decisions_to_regressed=analysis["wrong_decisions_to_regressed"])
        print("INTERFACE_REPAIR_V2_NEUTRAL_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failed = {"result_version": "hephaestus-interface-repair-v2-neutral-benchmark.v1", "status": "failed", "run_id": run_id, "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA"), "training_performed": False, "adapter_mutated": False, "error_type": type(exc).__name__, "error": str(exc), "completed_at_unix": time.time()}
        try:
            v2.put_json(client, result_key, failed)
        except Exception:
            pass
        raise
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
