#!/usr/bin/env python3
"""Run the paired raw-vs-normalized handoff experiment on the failed Phase II V1 pack.

Each producer output is generated once.  The same projected output is then presented
to the same consumer twice: once raw and once through the deterministic normalization
envelope.  This isolates presentation/interface effects before any weight update.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from hephaestus.evaluation.handoff_normalization import render_normalized_handoff
from hephaestus.evaluation.interface_mastery import score_interface_case, summarize_scorecards
from hephaestus.providers.models.role_stack import load_certified_role_model_stack

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_normalization_ab_v1.json"


def _load_v1_runner():
    path = ROOT / "scripts/run_interface_mastery_v1.py"
    spec = importlib.util.spec_from_file_location("interface_mastery_v1_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Phase II V1 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _put_json(client: Any, bucket: str, key: str, payload: object) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    client.put_object(Bucket=bucket, Key=key, Body=raw)
    response = client.get_object(Bucket=bucket, Key=key)
    try:
        observed = response["Body"].read()
    finally:
        response["Body"].close()
    if observed != raw:
        raise RuntimeError(f"S3 readback mismatch for {key}")


def _heartbeat(client: Any, bucket: str, prefix: str, stage: str, **extra: object) -> None:
    _put_json(client, bucket, f"{prefix}/heartbeat.json", {
        "stage": stage,
        "updated_at_unix": time.time(),
        **extra,
    })


def _compare(raw_summary: Mapping[str, Any], normalized_summary: Mapping[str, Any], rule: Mapping[str, Any]) -> dict[str, Any]:
    delta = {
        "overall_quality_100": float(normalized_summary["overall_quality_100"]) - float(raw_summary["overall_quality_100"]),
        "schema_compliance": float(normalized_summary["schema_compliance"]) - float(raw_summary["schema_compliance"]),
        "evidence_grounding": float(normalized_summary["evidence_grounding"]) - float(raw_summary["evidence_grounding"]),
        "interface_invariant_pass_rate": float(normalized_summary["interface_invariant_pass_rate"]) - float(raw_summary["interface_invariant_pass_rate"]),
        "consumer_exact_contract_pass_rate": float(normalized_summary["consumer_exact_contract_pass_rate"]) - float(raw_summary["consumer_exact_contract_pass_rate"]),
        "deterministic_boundary_pass_rate": float(normalized_summary["deterministic_boundary_pass_rate"]) - float(raw_summary["deterministic_boundary_pass_rate"]),
        "hallucination_rate": float(normalized_summary["hallucination_rate"]) - float(raw_summary["hallucination_rate"]),
    }
    checks = {
        "consumer_exact_gain": delta["consumer_exact_contract_pass_rate"] >= float(rule["minimum_consumer_exact_absolute_gain"]),
        "schema_no_regression": (not bool(rule["require_no_schema_regression"])) or delta["schema_compliance"] >= 0,
        "grounding_no_regression": (not bool(rule["require_no_evidence_grounding_regression"])) or delta["evidence_grounding"] >= 0,
        "boundary_no_regression": (not bool(rule["require_no_deterministic_boundary_regression"])) or delta["deterministic_boundary_pass_rate"] >= 0,
    }
    preferred = all(checks.values())
    return {
        "delta_normalized_minus_raw": delta,
        "decision_checks": checks,
        "normalization_preferred": preferred,
        "recommended_repair_presentation": "normalized_handoff" if preferred else "raw_handoff",
    }


def main() -> int:
    cfg = _load_json(CFG_PATH)
    source_cfg = _load_json(ROOT / cfg["source_phase_ii"]["config_path"])
    if source_cfg["pack"]["canonical_sha256"] != cfg["source_phase_ii"]["required_pack_sha256"]:
        raise RuntimeError("Phase II V1 pack identity drifted before normalization A/B")
    if cfg["governance"]["weights_mutation_allowed"]:
        raise RuntimeError("normalization A/B must remain inference-only")

    v1 = _load_v1_runner()
    _, pack = v1.load_pack(source_cfg)
    stack = load_certified_role_model_stack(ROOT / source_cfg["role_stack"]["registry_path"])
    if stack.automatic_role_dispatch_enabled:
        raise RuntimeError("automatic role dispatch must remain disabled during A/B")

    run_id = v1.required("HEPHAESTUS_INTERFACE_NORMALIZATION_AB_RUN_ID")
    repo_sha = v1.required("HEPHAESTUS_REPO_SHA")
    client = v1.s3_client()
    bucket = v1.bucket()
    prefix = f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    root = Path(f"/tmp/{run_id}")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "result_version": "hephaestus-interface-normalization-ab.v1",
        "protocol_id": cfg["protocol_id"],
        "run_id": run_id,
        "repo_sha": repo_sha,
        "source_pack_sha256": source_cfg["pack"]["canonical_sha256"],
        "source_failure_run_id": cfg["source_phase_ii"]["source_failure_run_id"],
        "status": "running",
        "weights_mutated": False,
        "production_promotion_performed": False,
        "automatic_role_dispatch_enabled": False,
        "started_at_unix": time.time(),
    }

    raw_scorecards = []
    normalized_scorecards = []
    seed = int(source_cfg["generation"]["seed"]) + 200000
    deadline = time.monotonic() + float(cfg["execution"]["hard_wall_seconds"]) - 60

    try:
        _heartbeat(client, bucket, prefix, "starting")
        for interface_index, interface_id in enumerate(source_cfg["interfaces"], 1):
            cases = pack["partitions"][interface_id]
            producer_role, consumer_role = interface_id.split("_to_")
            _heartbeat(client, bucket, prefix, "producer_loading", interface_id=interface_id)
            producer_model, producer_tokenizer, producer_runtime = v1.load_role_model(
                producer_role, stack, source_cfg, client, root / "assets"
            )
            upstream: dict[str, dict[str, Any]] = {}
            for case_index, case in enumerate(cases, 1):
                generation = v1.generate(
                    producer_model,
                    producer_tokenizer,
                    source_cfg["runtime"][producer_role],
                    v1.prompt_for_role(pack, case, producer_role),
                    seed + interface_index * 100 + case_index,
                    deadline,
                )
                upstream[str(case["case_id"])] = generation
            del producer_model, producer_tokenizer
            v1.release_cuda()
            gc.collect()

            _heartbeat(client, bucket, prefix, "consumer_loading", interface_id=interface_id)
            consumer_model, consumer_tokenizer, consumer_runtime = v1.load_role_model(
                consumer_role, stack, source_cfg, client, root / "assets"
            )

            for case_index, case in enumerate(cases, 1):
                producer_generation = upstream[str(case["case_id"])]
                producer_output = str(producer_generation["projected_output"])
                normalized_output = render_normalized_handoff(
                    source_role=producer_role,
                    raw_output=producer_output,
                )
                consumer_seed = seed + 10000 + interface_index * 100 + case_index
                raw_generation = v1.generate(
                    consumer_model,
                    consumer_tokenizer,
                    source_cfg["runtime"][consumer_role],
                    v1.prompt_for_role(pack, case, consumer_role, producer_output),
                    consumer_seed,
                    deadline,
                )
                normalized_generation = v1.generate(
                    consumer_model,
                    consumer_tokenizer,
                    source_cfg["runtime"][consumer_role],
                    v1.prompt_for_role(pack, case, consumer_role, normalized_output),
                    consumer_seed,
                    deadline,
                )

                raw_boundary_passed = True
                normalized_boundary_passed = True
                raw_boundary = None
                normalized_boundary = None
                if interface_id == "judge_to_controller":
                    raw_boundary_passed, raw_boundary = v1.controller_boundary(
                        case, str(raw_generation["projected_output"]), run_id
                    )
                    normalized_boundary_passed, normalized_boundary = v1.controller_boundary(
                        case, str(normalized_generation["projected_output"]), run_id
                    )

                raw_score = score_interface_case(
                    case=case,
                    producer_raw=producer_output,
                    consumer_raw=str(raw_generation["projected_output"]),
                    producer_vocabulary=pack["contract_vocabulary"][producer_role],
                    consumer_vocabulary=pack["contract_vocabulary"][consumer_role],
                    deterministic_boundary_passed=raw_boundary_passed,
                )
                normalized_score = score_interface_case(
                    case=case,
                    producer_raw=producer_output,
                    consumer_raw=str(normalized_generation["projected_output"]),
                    producer_vocabulary=pack["contract_vocabulary"][producer_role],
                    consumer_vocabulary=pack["contract_vocabulary"][consumer_role],
                    deterministic_boundary_passed=normalized_boundary_passed,
                )
                raw_scorecards.append(raw_score)
                normalized_scorecards.append(normalized_score)
                _put_json(client, bucket, f"{prefix}/samples/{interface_id}/{case_index:02d}-{case['case_id']}.json", {
                    "case": case,
                    "producer_generation": producer_generation,
                    "paired_handoff": {
                        "raw": producer_output,
                        "normalized": normalized_output,
                    },
                    "raw_arm": {
                        "consumer_generation": raw_generation,
                        "deterministic_boundary": raw_boundary,
                        "scorecard": raw_score.to_dict(),
                    },
                    "normalized_arm": {
                        "consumer_generation": normalized_generation,
                        "deterministic_boundary": normalized_boundary,
                        "scorecard": normalized_score.to_dict(),
                    },
                })
                _heartbeat(
                    client, bucket, prefix, "paired_case_complete",
                    interface_id=interface_id,
                    case_id=case["case_id"],
                    raw_quality=raw_score.quality_100,
                    normalized_quality=normalized_score.quality_100,
                )
            del consumer_model, consumer_tokenizer
            v1.release_cuda()
            gc.collect()

        raw_summary = summarize_scorecards(raw_scorecards, source_cfg["certification"], protocol_id=f"{cfg['protocol_id']}:raw")
        normalized_summary = summarize_scorecards(normalized_scorecards, source_cfg["certification"], protocol_id=f"{cfg['protocol_id']}:normalized")
        comparison = _compare(raw_summary, normalized_summary, cfg["decision_rule"])
        result.update({
            "status": "completed",
            "sample_count": len(raw_scorecards),
            "raw_summary": raw_summary,
            "normalized_summary": normalized_summary,
            "comparison": comparison,
            "completed_at_unix": time.time(),
        })
        _put_json(client, bucket, f"{prefix}/summary.json", {
            "raw": raw_summary,
            "normalized": normalized_summary,
            "comparison": comparison,
        })
        _put_json(client, bucket, f"{prefix}/result.json", result)
        _heartbeat(client, bucket, prefix, "complete", **comparison)
        print("INTERFACE_NORMALIZATION_AB_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        result.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "completed_at_unix": time.time(),
        })
        _put_json(client, bucket, f"{prefix}/result.json", result)
        print("INTERFACE_NORMALIZATION_AB_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        raise
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
