#!/usr/bin/env python3
"""Certify the frozen evaluator-v3.1-boundary-v1 candidate on sealed evidence.

This runner performs inference only. It never constructs an optimizer, never calls
backward, never saves or mutates an adapter, and never promotes production state.
A dedicated 8-case format preflight is evaluated first. The sealed 128-case
certification and 128-case regression partitions are touched only after preflight
passes. A certification claim is emitted only when every predeclared model and
system gate passes.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import run_interface_repair_v1 as v1
import run_interface_repair_v2 as v2
import run_interface_repair_v3_evaluator as v3
import run_interface_repair_v3_1_evaluator as v31
from interface_repair_v2_bootstrap import load_stack_compatible, materialize_parent_cached

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_evaluator_sealed_certification_v1.json"
ROLE = "evaluator"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def load_cfg() -> dict[str, Any]:
    cfg = load_json(CFG_PATH)
    governance = cfg["governance"]
    if governance.get("training_allowed") is not False:
        raise RuntimeError("sealed certification may not train")
    if governance.get("adapter_mutation_allowed") is not False:
        raise RuntimeError("sealed certification may not mutate adapters")
    if governance.get("candidate_weights_mutation_allowed") is not False:
        raise RuntimeError("sealed certification candidate weights must remain frozen")
    if governance.get("production_promotion_allowed") or governance.get("automatic_role_dispatch_allowed"):
        raise RuntimeError("sealed certification cannot promote or enable automatic dispatch")
    if governance.get("identical_rerun_after_scientific_failure_allowed"):
        raise RuntimeError("sealed certification may not allow blind identical reruns")
    return cfg


def load_candidate(cfg: Mapping[str, Any]) -> dict[str, Any]:
    candidate = load_json(ROOT / str(cfg["candidate_manifest"]))
    if candidate.get("candidate_id") != cfg["required_candidate_id"]:
        raise RuntimeError("candidate identity drift")
    if candidate.get("candidate_status") != "frozen_pending_fresh_sealed_certification":
        raise RuntimeError("candidate is not frozen pending sealed certification")
    governance = candidate["governance"]
    if governance.get("weights_frozen") is not True or governance.get("training_allowed") is not False:
        raise RuntimeError("candidate weights are not frozen")
    if governance.get("adapter_mutation_allowed") is not False:
        raise RuntimeError("candidate adapter mutation is not prohibited")
    sealed = candidate["sealed_evidence"]
    if sealed.get("status") != "frozen_unseen" or sealed.get("case_bodies_reviewed_after_freeze") is not False:
        raise RuntimeError("sealed evidence is no longer marked frozen and unseen")
    if sealed.get("canonical_sha256") != cfg["pack"]["canonical_sha256"]:
        raise RuntimeError("candidate/config sealed-pack SHA mismatch")
    return candidate


def verify_code_blobs(candidate: Mapping[str, Any]) -> dict[str, str]:
    mapping = {
        "evaluator_boundary.py": ROOT / "src/hephaestus/control/evaluator_boundary.py",
        "evaluator_evidence_refs.py": ROOT / "src/hephaestus/control/evaluator_evidence_refs.py",
        "run_interface_repair_v3_evaluator.py": ROOT / "scripts/run_interface_repair_v3_evaluator.py",
    }
    expected = candidate["architecture"]["code_blobs"]
    observed: dict[str, str] = {}
    for name, path in mapping.items():
        digest = subprocess.check_output(["git", "hash-object", str(path)], cwd=ROOT, text=True).strip()
        observed[name] = digest
        if digest != str(expected[name]):
            raise RuntimeError(f"frozen candidate code drift for {name}: {digest} != {expected[name]}")
    return observed


def load_pack(cfg: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    path = ROOT / str(cfg["pack"]["builder_path"])
    spec = importlib.util.spec_from_file_location("sealed_evaluator_pack", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import sealed certification builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pack = module.build_pack()
    module.validate(pack)
    observed = module.canonical_sha256(pack)
    expected = str(cfg["pack"]["canonical_sha256"])
    if observed != expected or observed != str(candidate["sealed_evidence"]["canonical_sha256"]):
        raise RuntimeError(f"sealed pack hash drift: {observed} != {expected}")
    return pack


def frozen_role_spec(cfg: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    stack = load_stack_compatible(cfg)
    source = dict(stack["roles"][ROLE])
    model = candidate["model"]
    if source.get("model_id") != model.get("model_id") or source.get("revision") != model.get("revision"):
        raise RuntimeError("candidate base model/revision differs from certified Phase-I evaluator base")
    source["adapter"] = {
        "s3_key": model["adapter_s3_key"],
        "sha256": model["adapter_sha256"],
        "bytes": int(model["adapter_bytes"]),
    }
    source["status"] = "frozen_pending_sealed_certification"
    return source


def _model_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(cfg))
    clone["training"] = {
        "lora_rank": int(cfg["model_protocol"]["lora_rank"]),
        "lora_alpha": int(cfg["model_protocol"]["lora_alpha"]),
    }
    return clone


def _put_jsonl(client: Any, key: str, rows: list[dict[str, Any]]) -> None:
    raw = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    client.put_object(Bucket=v1.bucket(), Key=key, Body=raw, ContentType="application/x-ndjson")
    response = client.get_object(Bucket=v1.bucket(), Key=key)
    try:
        observed = response["Body"].read()
    finally:
        response["Body"].close()
    if observed != raw:
        raise RuntimeError(f"S3 readback mismatch for {key}")


def evaluate_partition(
    *,
    cfg: Mapping[str, Any],
    pack: Mapping[str, Any],
    partition: str,
    model: Any,
    tokenizer: Any,
    deadline: float,
    client: Any,
    prefix: str,
    heartbeat,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cases = list(pack["partitions"][ROLE][partition])
    runtime = dict(cfg["roles"][ROLE])
    seed = int(cfg["model_protocol"]["seed"])
    shard_size = int(cfg["execution"]["evaluation_shard_size"])
    rows: list[dict[str, Any]] = []
    shard: list[dict[str, Any]] = []
    shard_index = 0
    heartbeat(f"{partition}_started", sample_count=len(cases))
    for index, case in enumerate(cases, 1):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"sealed certification hard wall reached during {partition}")
        text = v1.prompt(pack, case, ROLE)
        generation = v31.generate_v31(model, tokenizer, runtime, text, seed + index + (10000 if partition == "regression" else 0), deadline)
        score = v3.score_case_v3(pack, case, ROLE, str(generation["projected_output"]))
        row = {
            "case_id": case["case_id"],
            "role": ROLE,
            "partition": partition,
            "kind": case.get("kind"),
            "semantic_root": case.get("semantic_root"),
            "generation": generation,
            "score": score,
            "expected": case["expected"],
        }
        rows.append(row)
        shard.append(row)
        if len(shard) >= shard_size or index == len(cases):
            shard_index += 1
            _put_jsonl(client, f"{prefix}/samples/{partition}/shard-{shard_index:04d}.jsonl", shard)
            shard = []
        if index == 1 or index % 32 == 0 or index == len(cases):
            partial = v31.summarize_v31(rows)
            heartbeat(
                f"{partition}_progress",
                completed=index,
                total=len(cases),
                semantic_quality_100=partial["semantic_quality_100"],
                decision_exact_pass_rate=partial["decision_exact_pass_rate"],
                system_action_exact_pass_rate=partial["system_action_exact_pass_rate"],
            )
    summary = v31.summarize_v31(rows)
    return rows, summary


def decision_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: Counter[str] = Counter()
    total: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in rows:
        expected = str(row["expected"]["decision"])
        parsed = row["score"].get("parsed")
        predicted = str(parsed.get("decision", "<invalid>")) if isinstance(parsed, dict) else "<invalid>"
        confusion[f"{expected} -> {predicted}"] += 1
        total[expected] += 1
        if expected == predicted:
            correct[expected] += 1
    return {
        "decision_confusion": dict(sorted(confusion.items())),
        "per_state_decision_accuracy": {
            state: correct[state] / max(1, total[state]) for state in sorted(total)
        },
        "wrong_decision_count": sum(total.values()) - sum(correct.values()),
    }


def certify(
    cfg: Mapping[str, Any],
    cert_summary: Mapping[str, float],
    regression_summary: Mapping[str, float],
) -> dict[str, Any]:
    model = cfg["certification"]["model"]
    system = cfg["certification"]["system"]
    regression_drop = float(cert_summary["semantic_quality_100"]) - float(regression_summary["semantic_quality_100"])
    model_checks = {
        "certification_semantic_quality": cert_summary["semantic_quality_100"] >= float(model["minimum_certification_semantic_quality_100"]),
        "certification_semantic_exact": cert_summary["semantic_exact_pass_rate"] >= float(model["minimum_certification_semantic_exact_pass_rate"]),
        "regression_semantic_quality": regression_summary["semantic_quality_100"] >= float(model["minimum_regression_semantic_quality_100"]),
        "regression_semantic_exact": regression_summary["semantic_exact_pass_rate"] >= float(model["minimum_regression_semantic_exact_pass_rate"]),
        "certification_schema": cert_summary["schema_compliance"] >= float(model["schema_compliance_required"]),
        "regression_schema": regression_summary["schema_compliance"] >= float(model["schema_compliance_required"]),
        "certification_grounding": cert_summary["evidence_grounding"] >= float(model["minimum_evidence_grounding"]),
        "regression_grounding": regression_summary["evidence_grounding"] >= float(model["minimum_evidence_grounding"]),
        "certification_hallucination": cert_summary["hallucination_rate"] <= float(model["maximum_hallucination_rate"]),
        "regression_hallucination": regression_summary["hallucination_rate"] <= float(model["maximum_hallucination_rate"]),
        "certification_to_regression_quality_drop": regression_drop <= float(model["maximum_certification_to_regression_semantic_quality_drop_100"]),
    }
    system_checks = {
        "certification_primary_projection": cert_summary["primary_variable_projection_rate"] >= float(system["primary_variable_projection_rate_required"]),
        "regression_primary_projection": regression_summary["primary_variable_projection_rate"] >= float(system["primary_variable_projection_rate_required"]),
        "certification_boundary_projection": cert_summary["boundary_projection_rate"] >= float(system["boundary_projection_rate_required"]),
        "regression_boundary_projection": regression_summary["boundary_projection_rate"] >= float(system["boundary_projection_rate_required"]),
        "certification_system_action_exact": cert_summary["system_action_exact_pass_rate"] >= float(system["system_action_exact_pass_rate_required"]),
        "regression_system_action_exact": regression_summary["system_action_exact_pass_rate"] >= float(system["system_action_exact_pass_rate_required"]),
        "certification_no_system_escalation": cert_summary["system_semantic_escalation_rate"] <= float(system["maximum_system_semantic_escalation_rate"]),
        "regression_no_system_escalation": regression_summary["system_semantic_escalation_rate"] <= float(system["maximum_system_semantic_escalation_rate"]),
    }
    model_passed = all(model_checks.values())
    system_passed = all(system_checks.values())
    return {
        "certified": model_passed and system_passed,
        "model_certification": {"passed": model_passed, "checks": model_checks},
        "system_certification": {"passed": system_passed, "checks": system_checks},
        "certification_to_regression_semantic_quality_drop_100": regression_drop,
    }


def main() -> int:
    cfg = load_cfg()
    candidate = load_candidate(cfg)
    run_id = (os.environ.get("HEPHAESTUS_EVALUATOR_CERTIFICATION_RUN_ID") or "").strip()
    if not run_id:
        raise RuntimeError("HEPHAESTUS_EVALUATOR_CERTIFICATION_RUN_ID is required")
    prefix = f"{str(cfg['execution']['s3_prefix']).rstrip('/')}/{run_id}"
    progress_key = f"{prefix}/progress.json"
    result_key = f"{prefix}/result.json"
    client = v1.s3_client()
    client.head_bucket(Bucket=v1.bucket())
    deadline = time.monotonic() + float(cfg["execution"]["hard_wall_seconds"])
    started_at = time.time()
    touched_sealed = False

    def heartbeat(stage: str, **extra: Any) -> None:
        payload = {
            "run_id": run_id,
            "stage": stage,
            "timestamp_unix": time.time(),
            "remaining_wall_seconds": max(0.0, deadline - time.monotonic()),
            **extra,
        }
        v1.put_json(client, progress_key, payload)
        print("EVALUATOR_SEALED_CERT_PROGRESS_JSON " + json.dumps(payload, sort_keys=True), flush=True)

    try:
        heartbeat("candidate_verification_started")
        observed_blobs = verify_code_blobs(candidate)
        pack = load_pack(cfg, candidate)
        role_spec = frozen_role_spec(cfg, candidate)
        root = Path("/opt/hephaestus-evaluator-sealed-certification")
        root.mkdir(parents=True, exist_ok=True)
        heartbeat("materializing_frozen_candidate", adapter_sha256=role_spec["adapter"]["sha256"])
        base_dir, adapter_dir = materialize_parent_cached(client, role_spec, root)
        model, tokenizer, runtime_info = v1.load_model(ROLE, _model_cfg(cfg), role_spec, base_dir, adapter_dir)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_after_freeze != 0:
            raise RuntimeError("sealed certification unexpectedly exposes trainable parameters")

        preflight_rows, preflight_summary = evaluate_partition(
            cfg=cfg, pack=pack, partition=str(cfg["pack"]["preflight_partition"]),
            model=model, tokenizer=tokenizer, deadline=deadline, client=client,
            prefix=prefix, heartbeat=heartbeat,
        )
        preflight_passed = (
            preflight_summary["final_complete_contract_json_rate"] >= float(cfg["preflight"]["require_complete_contract_json_rate"])
            and preflight_summary["schema_compliance"] >= float(cfg["preflight"]["require_schema_compliance"])
        )
        preflight_record = {
            "passed": preflight_passed,
            "sample_count": len(preflight_rows),
            "final_complete_contract_json_rate": preflight_summary["final_complete_contract_json_rate"],
            "initial_complete_contract_json_rate": preflight_summary["initial_complete_contract_json_rate"],
            "schema_compliance": preflight_summary["schema_compliance"],
            "format_retry_attempt_rate": preflight_summary["format_retry_attempt_rate"],
            "format_retry_success_rate": preflight_summary["format_retry_success_rate"],
        }
        if not preflight_passed:
            result = {
                "result_version": "hephaestus-evaluator-sealed-certification.v1",
                "status": "preflight_rejected",
                "certified": False,
                "run_id": run_id,
                "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA"),
                "candidate_id": candidate["candidate_id"],
                "candidate_adapter_sha256": candidate["model"]["adapter_sha256"],
                "pack_sha256": cfg["pack"]["canonical_sha256"],
                "training_performed": False,
                "adapter_mutated": False,
                "sealed_certification_partitions_touched": False,
                "certification_claim_performed": False,
                "production_promotion_performed": False,
                "preflight": preflight_record,
                "runtime": {**runtime_info, "trainable_parameter_count_after_freeze": 0},
                "completed_at_unix": time.time(),
            }
            v1.put_json(client, result_key, result)
            print("EVALUATOR_SEALED_CERT_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
            return 3

        heartbeat("sealed_certification_begin")
        touched_sealed = True
        cert_rows, cert_summary = evaluate_partition(
            cfg=cfg, pack=pack, partition=str(cfg["pack"]["certification_partition"]),
            model=model, tokenizer=tokenizer, deadline=deadline, client=client,
            prefix=prefix, heartbeat=heartbeat,
        )
        regression_rows, regression_summary = evaluate_partition(
            cfg=cfg, pack=pack, partition=str(cfg["pack"]["regression_partition"]),
            model=model, tokenizer=tokenizer, deadline=deadline, client=client,
            prefix=prefix, heartbeat=heartbeat,
        )
        certification = certify(cfg, cert_summary, regression_summary)
        certified = bool(certification["certified"])
        result = {
            "result_version": "hephaestus-evaluator-sealed-certification.v1",
            "status": "certified" if certified else "scientifically_rejected",
            "certified": certified,
            "run_id": run_id,
            "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA"),
            "candidate_id": candidate["candidate_id"],
            "candidate_adapter_sha256": candidate["model"]["adapter_sha256"],
            "base_model_id": candidate["model"]["model_id"],
            "base_model_revision": candidate["model"]["revision"],
            "pack_sha256": cfg["pack"]["canonical_sha256"],
            "observed_code_blobs": observed_blobs,
            "training_performed": False,
            "adapter_mutated": False,
            "adapter_persisted": False,
            "sealed_certification_partitions_touched": True,
            "certification_claim_performed": certified,
            "production_promotion_performed": False,
            "automatic_role_dispatch_performed": False,
            "preflight": preflight_record,
            "certification_summary": cert_summary,
            "regression_summary": regression_summary,
            "certification_analysis": decision_analysis(cert_rows),
            "regression_analysis": decision_analysis(regression_rows),
            "certification": certification,
            "runtime": {**runtime_info, "trainable_parameter_count_after_freeze": 0},
            "started_at_unix": started_at,
            "completed_at_unix": time.time(),
        }
        v1.put_json(client, result_key, result)
        heartbeat(
            "complete",
            certified=certified,
            certification_semantic_exact=cert_summary["semantic_exact_pass_rate"],
            regression_semantic_exact=regression_summary["semantic_exact_pass_rate"],
            certification_system_action_exact=cert_summary["system_action_exact_pass_rate"],
            regression_system_action_exact=regression_summary["system_action_exact_pass_rate"],
        )
        print("EVALUATOR_SEALED_CERT_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        return 0 if certified else 2
    except Exception as exc:
        failed = {
            "result_version": "hephaestus-evaluator-sealed-certification.v1",
            "status": "failed",
            "certified": False,
            "run_id": run_id,
            "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA"),
            "candidate_id": candidate.get("candidate_id"),
            "pack_sha256": cfg["pack"]["canonical_sha256"],
            "training_performed": False,
            "adapter_mutated": False,
            "sealed_certification_partitions_touched": touched_sealed,
            "certification_claim_performed": False,
            "production_promotion_performed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_at_unix": time.time(),
        }
        try:
            v1.put_json(client, result_key, failed)
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
