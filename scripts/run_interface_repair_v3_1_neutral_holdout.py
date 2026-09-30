#!/usr/bin/env python3
"""Run the frozen V3.1 Evaluator on the paired neutral-identifier holdout.

No optimizer is constructed, no backward pass occurs, no adapter is saved, and no
certification or production-promotion claim is permitted. The experiment asks one
causal question: does replacing model-visible evaluation namespace identifiers with
opaque identifiers materially rescue the frozen V3.1 adapter?
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import run_interface_repair_v1 as v1
import run_interface_repair_v2 as v2
import run_interface_repair_v3_evaluator as v3
import run_interface_repair_v3_1_evaluator as v31
from interface_repair_v2_bootstrap import load_stack_compatible, materialize_parent_cached

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_repair_v3_1_neutral_holdout.json"
ROLE = "evaluator"
PARTITION = "neutral_holdout"


def load_cfg() -> dict[str, Any]:
    value = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("neutral holdout config must be an object")
    if value.get("diagnostic_only") is not True or value.get("training_performed") is not False:
        raise RuntimeError("neutral holdout must remain diagnostic-only and training-free")
    governance = value["governance"]
    if governance.get("training_allowed") is not False or governance.get("adapter_mutation_allowed") is not False:
        raise RuntimeError("neutral holdout may not train or mutate an adapter")
    if governance.get("production_promotion_allowed") or governance.get("automatic_role_dispatch_allowed"):
        raise RuntimeError("neutral holdout cannot promote or dispatch")
    if governance.get("certification_claim_allowed"):
        raise RuntimeError("neutral holdout cannot make a certification claim")
    return value


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
    expected = str(cfg["pack"]["canonical_sha256"])
    if observed != expected:
        raise RuntimeError(f"neutral holdout pack hash drift: {observed} != {expected}")
    return pack


def load_frozen_stack(cfg: Mapping[str, Any]) -> dict[str, Any]:
    stack = load_stack_compatible(cfg)
    source = dict(cfg["source_v31_adapter"])
    if source.get("status") != "frozen_scientifically_rejected_v3_1_candidate":
        raise RuntimeError("neutral diagnostic source adapter status drift")
    evaluator = dict(stack["roles"][ROLE])
    if evaluator.get("model_id") != source.get("model_id") or evaluator.get("revision") != source.get("revision"):
        raise RuntimeError("neutral diagnostic model/revision differs from the V3.1 source")
    evaluator["adapter"] = {
        "s3_key": source["s3_key"],
        "sha256": source["sha256"],
        "bytes": int(source["bytes"]),
    }
    evaluator["status"] = "frozen_v3_1_neutral_diagnostic"
    roles = dict(stack["roles"])
    roles[ROLE] = evaluator
    out = dict(stack)
    out["roles"] = roles
    return out


def _model_load_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(cfg))
    clone["training"] = {
        "lora_rank": int(cfg["model_protocol"]["lora_rank"]),
        "lora_alpha": int(cfg["model_protocol"]["lora_alpha"]),
    }
    return clone


def _analysis(rows: list[dict[str, Any]], summary: Mapping[str, float], cfg: Mapping[str, Any]) -> dict[str, Any]:
    confusion: Counter[str] = Counter()
    state_total: Counter[str] = Counter()
    state_correct: Counter[str] = Counter()
    wrong = 0
    wrong_to_regressed = 0
    primary_confusion: Counter[str] = Counter()

    for row in rows:
        expected = str(row["expected"]["decision"])
        expected_primary = str(row["expected"]["primary_variable"])
        parsed = row["score"].get("parsed")
        predicted = str(parsed.get("decision", "<invalid>")) if isinstance(parsed, dict) else "<invalid>"
        predicted_primary = str(parsed.get("primary_variable", "<invalid>")) if isinstance(parsed, dict) else "<invalid>"
        confusion[f"{expected} -> {predicted}"] += 1
        primary_confusion[f"{expected_primary} -> {predicted_primary}"] += 1
        state_total[expected] += 1
        if predicted == expected:
            state_correct[expected] += 1
        else:
            wrong += 1
            if predicted == "regressed":
                wrong_to_regressed += 1

    gates = cfg["diagnostic_gates"]
    checks = {
        "semantic_quality": float(summary["semantic_quality_100"]) >= float(gates["minimum_semantic_quality_100"]),
        "semantic_exact": float(summary["semantic_exact_pass_rate"]) >= float(gates["minimum_semantic_exact_pass_rate"]),
        "decision_exact": float(summary["decision_exact_pass_rate"]) >= float(gates["minimum_decision_exact_pass_rate"]),
        "schema": float(summary["schema_compliance"]) >= float(gates["schema_compliance_required"]),
        "grounding": float(summary["evidence_grounding"]) >= float(gates["minimum_evidence_grounding"]),
        "hallucination": float(summary["hallucination_rate"]) <= float(gates["maximum_hallucination_rate"]),
        "boundary_projection": float(summary["boundary_projection_rate"]) >= float(gates["boundary_projection_rate_required"]),
        "system_action_exact": float(summary["system_action_exact_pass_rate"]) >= float(gates["system_action_exact_pass_rate_required"]),
        "no_system_escalation": float(summary["system_semantic_escalation_rate"]) <= float(gates["maximum_system_semantic_escalation_rate"]),
    }
    reference = cfg["source_reference"]
    quality_delta = float(summary["semantic_quality_100"]) - float(reference["semantic_quality_100"])
    exact_delta = float(summary["semantic_exact_pass_rate"]) - float(reference["semantic_exact_pass_rate"])
    decision_delta = float(summary["decision_exact_pass_rate"]) - float(reference["decision_exact_pass_rate"])

    return {
        "diagnostic_pass": all(checks.values()),
        "checks": checks,
        "paired_reference": dict(reference),
        "delta_vs_v31_post_regression": {
            "semantic_quality_100": quality_delta,
            "semantic_exact_pass_rate": exact_delta,
            "decision_exact_pass_rate": decision_delta,
        },
        "decision_confusion": dict(sorted(confusion.items())),
        "primary_variable_confusion": dict(sorted(primary_confusion.items())),
        "per_state_decision_accuracy": {
            state: state_correct[state] / max(1, state_total[state]) for state in sorted(state_total)
        },
        "wrong_decision_count": wrong,
        "wrong_decisions_to_regressed": wrong_to_regressed,
        "wrong_decisions_to_regressed_fraction": wrong_to_regressed / max(1, wrong),
        "interpretation": (
            "supports_material_identifier_leakage_hypothesis"
            if all(checks.values())
            else "neutral_identifiers_do_not_rescue_v31_to_predeclared_gates"
        ),
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
    started_wall = time.time()
    deadline = time.monotonic() + float(cfg["execution"]["hard_wall_seconds"])

    def heartbeat(stage: str, **extra: Any) -> None:
        payload = {
            "run_id": run_id,
            "stage": stage,
            "timestamp_unix": time.time(),
            "remaining_wall_seconds": max(0.0, deadline - time.monotonic()),
            **extra,
        }
        v2.put_json(client, progress_key, payload)
        print("INTERFACE_REPAIR_V3_NEUTRAL_PROGRESS_JSON " + json.dumps(payload, sort_keys=True), flush=True)

    try:
        heartbeat("pack_validation_started")
        pack = load_pack(cfg)
        cases = list(pack["partitions"][ROLE][PARTITION])
        if len(cases) != int(cfg["pack"]["cases"]):
            raise RuntimeError("neutral holdout case-count drift")

        stack = load_frozen_stack(cfg)
        role_spec = stack["roles"][ROLE]
        root = Path("/opt/hephaestus-neutral-holdout")
        root.mkdir(parents=True, exist_ok=True)
        heartbeat("materializing_frozen_v31_adapter", adapter_sha256=role_spec["adapter"]["sha256"])
        base_dir, adapter_dir = materialize_parent_cached(client, role_spec, root)
        model, tokenizer, runtime_info = v1.load_model(
            ROLE, _model_load_cfg(cfg), role_spec, base_dir, adapter_dir
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_after_freeze != 0:
            raise RuntimeError("neutral diagnostic unexpectedly exposes trainable parameters")

        runtime = dict(cfg["roles"][ROLE])
        rows: list[dict[str, Any]] = []
        shard: list[dict[str, Any]] = []
        shard_index = 0
        seed = int(cfg["model_protocol"]["seed"])
        shard_size = int(cfg["execution"]["evaluation_shard_size"])
        heartbeat("neutral_evaluation_started", cases=len(cases), trainable_parameters=0)

        for i, case in enumerate(cases, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("neutral holdout hard wall reached during evaluation")
            text = v1.prompt(pack, case, ROLE)
            generation = v31.generate_v31(model, tokenizer, runtime, text, seed + i, deadline)
            score = v3.score_case_v3(pack, case, ROLE, str(generation["projected_output"]))
            row = {
                "case_id": case["case_id"],
                "source_case_id": case["source_case_id"],
                "role": ROLE,
                "phase": PARTITION,
                "kind": case.get("kind"),
                "context": case.get("context"),
                "semantic_root": case.get("semantic_root"),
                "generation": generation,
                "score": score,
                "expected": case["expected"],
            }
            rows.append(row)
            shard.append(row)
            if len(shard) >= shard_size or i == len(cases):
                shard_index += 1
                v2.put_jsonl(client, f"{prefix}/samples/shard-{shard_index:04d}.jsonl", shard)
                shard = []
            if i == 1 or i % 32 == 0 or i == len(cases):
                partial = v31.summarize_v31(rows)
                heartbeat(
                    "neutral_evaluation_progress",
                    completed=i,
                    total=len(cases),
                    semantic_quality_100=partial["semantic_quality_100"],
                    decision_exact_pass_rate=partial["decision_exact_pass_rate"],
                )

        summary = v31.summarize_v31(rows)
        analysis = _analysis(rows, summary, cfg)
        result = {
            "result_version": "hephaestus-interface-repair-v3.1-neutral-holdout.v1",
            "status": "completed",
            "run_id": run_id,
            "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA"),
            "pack_sha256": cfg["pack"]["canonical_sha256"],
            "source_v31_adapter": dict(cfg["source_v31_adapter"]),
            "training_performed": False,
            "adapter_mutated": False,
            "adapter_persisted": False,
            "certification_claim_performed": False,
            "production_promotion_performed": False,
            "automatic_role_dispatch_performed": False,
            "runtime": {
                **runtime_info,
                "trainable_parameter_count_after_freeze": trainable_after_freeze,
            },
            "summary": summary,
            "analysis": analysis,
            "started_at_unix": started_wall,
            "completed_at_unix": time.time(),
        }
        v2.put_json(client, result_key, result)
        heartbeat(
            "complete",
            diagnostic_pass=analysis["diagnostic_pass"],
            semantic_quality_100=summary["semantic_quality_100"],
            semantic_exact_pass_rate=summary["semantic_exact_pass_rate"],
            wrong_decisions_to_regressed=analysis["wrong_decisions_to_regressed"],
        )
        print("INTERFACE_REPAIR_V3_NEUTRAL_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failed = {
            "result_version": "hephaestus-interface-repair-v3.1-neutral-holdout.v1",
            "status": "failed",
            "run_id": run_id,
            "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA"),
            "training_performed": False,
            "adapter_mutated": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_at_unix": time.time(),
        }
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
