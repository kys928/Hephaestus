#!/usr/bin/env python3
"""Runtime bridge for distance-to-role V1 FP16 topology -> BF16 training dose-zero verification."""
from __future__ import annotations

import copy
import math
from typing import Any

import run_distance_to_role_v1 as dtr

_ORIGINAL_TRAIN_ROLE_CURVE = dtr.train_role_curve
_SOURCE_RUNTIME_BY_MODEL: dict[str, dict[str, Any]] = {}


def classify_precision_bridge(source_dtypes: list[str], trajectory_dtypes: list[str]) -> str:
    source = sorted(map(str, source_dtypes))
    trajectory = sorted(map(str, trajectory_dtypes))
    if source == trajectory:
        return "same_precision"
    if source == ["torch.float16"] and trajectory == ["torch.bfloat16"]:
        return "fp16_topology_to_bf16_training"
    return "unsupported_precision_bridge"


def floating_dtypes(model: Any) -> list[str]:
    return sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()})


def bridged_train_role_curve(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
    base_model = kwargs["base_model"]
    tokenizer = kwargs["tokenizer"]
    candidate = kwargs["candidate"]
    role = kwargs["role"]
    topology = kwargs["topology"]
    protocol = kwargs["protocol"]
    work_dir = kwargs["work_dir"]

    source_role = kwargs["baseline_topology_role"]
    source_quality = float(source_role["quality_100"])
    source_runtime = _SOURCE_RUNTIME_BY_MODEL.get(candidate["model_id"], {})
    source_dtypes = [str(value) for value in source_runtime.get("weight_dtypes", [])]
    trajectory_dtypes = floating_dtypes(base_model)
    bridge = classify_precision_bridge(source_dtypes, trajectory_dtypes)

    screening_seeds = [int(value) for value in protocol["role_screening"]["seeds"]]
    probe = dtr.evaluate_role(base_model, tokenizer, topology, candidate, role, screening_seeds, None)
    trajectory_quality = float(probe["summary"]["quality_100"])

    if bridge == "same_precision" and not math.isclose(trajectory_quality, source_quality, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(
            f"unexplained same-precision dose-zero topology drift for {candidate['model_id']} role={role}: "
            f"{trajectory_quality} != {source_quality}"
        )
    if bridge == "unsupported_precision_bridge":
        raise RuntimeError(
            f"unsupported dose-zero precision bridge for {candidate['model_id']} role={role}: "
            f"source={source_dtypes} trajectory={trajectory_dtypes}"
        )

    patched_source_role = copy.deepcopy(source_role)
    patched_source_role["quality_100"] = trajectory_quality
    patched_kwargs = dict(kwargs)
    patched_kwargs["baseline_topology_role"] = patched_source_role

    returned_model, curve = _ORIGINAL_TRAIN_ROLE_CURVE(**patched_kwargs)
    actual_zero = float(curve["points"][0]["role"]["quality_100"])
    if not math.isclose(actual_zero, trajectory_quality, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(
            f"dose-zero bridge probe was not reproducible for {candidate['model_id']} role={role}: "
            f"probe={trajectory_quality} recorded={actual_zero}"
        )

    curve["baseline_topology_quality_100"] = source_quality
    curve["trajectory_dose_zero_quality_100"] = trajectory_quality
    curve["dose_zero_bridge"] = {
        "classification": bridge,
        "source_topology_quality_100": source_quality,
        "trajectory_quality_100": trajectory_quality,
        "quality_delta_100": trajectory_quality - source_quality,
        "source_weight_dtypes": source_dtypes,
        "trajectory_weight_dtypes": trajectory_dtypes,
        "source_gpu": source_runtime.get("gpu"),
        "reason": "Test 1 topology used FP16 inference while the frozen Test 3 LoRA trajectory uses BF16 training precision; the fresh BF16 dose-zero point is the internal trajectory baseline.",
    }
    dtr.atomic_json(work_dir / "role_curve.json", curve)
    return returned_model, curve


def main() -> int:
    protocol, _, _, topology_sha, _, _ = dtr.load_protocols()
    baseline = dtr.elastic._load_baseline(
        {"baseline": {"run_id": protocol["sources"]["topology_run_id"]}},
        topology_sha,
    )
    _SOURCE_RUNTIME_BY_MODEL.clear()
    for row in baseline.get("model_summaries", []):
        model_id = str(row.get("model_id", ""))
        runtime = row.get("runtime")
        if model_id and isinstance(runtime, dict):
            _SOURCE_RUNTIME_BY_MODEL[model_id] = runtime
    dtr.train_role_curve = bridged_train_role_curve
    return dtr.main()


if __name__ == "__main__":
    raise SystemExit(main())
