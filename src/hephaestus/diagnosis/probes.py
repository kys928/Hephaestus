"""Measured post-failure probes for resolving an inconclusive diagnosis.

The probes emit only signals supported by recorded measurements.  They do not
execute interventions and they do not turn a regression into a causal claim.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PostFailureProbePolicy:
    """Conservative thresholds for evidence generation, not causal certainty."""

    maximum_budget_epoch_fraction: float = 0.05
    minimum_total_loss_drop_fraction: float = 0.05
    maximum_improving_tail_slope_per_step: float = -0.001
    minimum_tail_points: int = 4
    scheduler_tolerance: float = 1e-9


@dataclass(slots=True)
class PostFailureProbeResult:
    evidence_records: list[dict[str, object]] = field(default_factory=list)
    measurements: dict[str, object] = field(default_factory=dict)
    unresolved_questions: list[str] = field(default_factory=list)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _linear_slope(points: Sequence[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator <= 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def analyze_training_dynamics(
    metrics: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    source_ref: str,
    policy: PostFailureProbePolicy | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Measure bounded-run learning dynamics and scheduler conformance."""

    policy = policy or PostFailureProbePolicy()
    rows = sorted((dict(row) for row in metrics), key=lambda row: int(row.get("step", 0)))
    if len(rows) < 2:
        raise ValueError("at least two metric records are required")

    required = ("step", "training_loss", "learning_rate", "gradient_norm", "epoch")
    for index, row in enumerate(rows):
        for key in required:
            if key not in row or not _finite(row[key]):
                raise ValueError(f"metrics[{index}] missing finite {key}")

    steps = [int(row["step"]) for row in rows]
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("metric steps must be strictly increasing")

    losses = [float(row["training_loss"]) for row in rows]
    lrs = [float(row["learning_rate"]) for row in rows]
    gradients = [float(row["gradient_norm"]) for row in rows]
    epochs = [float(row["epoch"]) for row in rows]
    maximum_steps = int(config.get("max_steps", 0) or 0)
    warmup_steps = int(config.get("warmup_steps", 0) or 0)
    base_lr = float(config.get("learning_rate", 0.0) or 0.0)
    scheduler = str(config.get("scheduler", ""))

    tail_count = min(len(rows), max(policy.minimum_tail_points, len(rows) // 3))
    tail = list(zip(steps[-tail_count:], losses[-tail_count:]))
    tail_slope = _linear_slope([(float(x), y) for x, y in tail])
    total_drop = losses[0] - losses[-1]
    drop_fraction = total_drop / max(abs(losses[0]), 1e-12)

    finite_dynamics = all(_finite(value) for value in [*losses, *lrs, *gradients, *epochs])
    nonnegative_lrs = all(value >= -policy.scheduler_tolerance for value in lrs)
    within_base_lr = base_lr <= 0 or all(value <= base_lr + policy.scheduler_tolerance for value in lrs)

    warmup_pairs = [(step, lr) for step, lr in zip(steps, lrs) if step <= warmup_steps]
    decay_pairs = [(step, lr) for step, lr in zip(steps, lrs) if step >= warmup_steps]
    warmup_monotone = all(
        right + policy.scheduler_tolerance >= left
        for (_, left), (_, right) in zip(warmup_pairs, warmup_pairs[1:])
    )
    decay_monotone = all(
        right <= left + policy.scheduler_tolerance
        for (_, left), (_, right) in zip(decay_pairs, decay_pairs[1:])
    )
    final_lr_near_zero = (
        scheduler != "linear"
        or maximum_steps <= 0
        or steps[-1] < maximum_steps
        or abs(lrs[-1]) <= max(policy.scheduler_tolerance, abs(base_lr) * 0.01)
    )
    scheduler_conforms = (
        finite_dynamics
        and nonnegative_lrs
        and within_base_lr
        and warmup_monotone
        and decay_monotone
        and final_lr_near_zero
    )

    reached_step_budget = maximum_steps > 0 and steps[-1] >= maximum_steps
    tiny_epoch_fraction = epochs[-1] <= policy.maximum_budget_epoch_fraction
    loss_materially_lower = drop_fraction >= policy.minimum_total_loss_drop_fraction
    tail_still_improving = (
        tail_slope is not None
        and tail_slope <= policy.maximum_improving_tail_slope_per_step
    )
    undertraining_supported = (
        reached_step_budget
        and tiny_epoch_fraction
        and loss_materially_lower
        and tail_still_improving
        and finite_dynamics
    )

    measurements: dict[str, object] = {
        "metric_points": len(rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        "configured_max_steps": maximum_steps,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "total_loss_drop": total_drop,
        "total_loss_drop_fraction": drop_fraction,
        "tail_point_count": tail_count,
        "tail_loss_slope_per_step": tail_slope,
        "final_epoch_fraction": epochs[-1],
        "minimum_gradient_norm": min(gradients),
        "maximum_gradient_norm": max(gradients),
        "base_learning_rate": base_lr,
        "final_learning_rate": lrs[-1],
        "scheduler": scheduler,
        "warmup_steps": warmup_steps,
        "scheduler_conforms_to_recorded_shape": scheduler_conforms,
        "reached_configured_step_budget": reached_step_budget,
        "tiny_epoch_fraction": tiny_epoch_fraction,
        "loss_materially_lower": loss_materially_lower,
        "tail_still_improving": tail_still_improving,
    }
    evidence = {
        "evidence_kind": "training_metrics",
        "source_ref": source_ref,
        "summary": (
            "Measured the complete persisted optimization trace for budget exhaustion, "
            "learning-curve direction, numerical stability, and scheduler conformance."
        ),
        "confidence": 0.95,
        "numerically_stable": finite_dynamics,
        "optimizer_stable": finite_dynamics and scheduler_conforms,
        "scheduler_misconfigured": not scheduler_conforms,
        "training_budget_exhausted": undertraining_supported,
        "undertraining_detected": undertraining_supported,
        "measurements": measurements,
    }
    return evidence, measurements


def analyze_dataset_task_coverage(
    rows: Iterable[Mapping[str, object]],
    eval_pack: Mapping[str, object],
    *,
    source_ref: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Measure whether training records structurally cover frozen eval task forms."""

    total = 0
    text_rows = 0
    prompt_target_rows = 0
    structured_target_rows = 0
    exact_prompt_hits = 0
    instruction_cue_rows = 0

    prompts: list[str] = []
    for collection in (
        "generation_probes",
        "continuation_prompts",
        "structure_tests",
        "repetition_checks",
        "length_termination_checks",
    ):
        values = eval_pack.get(collection, [])
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict) and str(item.get("prompt", "")).strip():
                    prompts.append(str(item["prompt"]).strip().lower())

    instruction_cues = (
        "reply with exactly",
        "return json",
        "in one short sentence",
        "in one sentence",
        "at most",
    )

    for raw in rows:
        row = dict(raw)
        total += 1
        text = str(row.get("text", ""))
        if text:
            text_rows += 1
            lower = text.lower()
            if any(prompt in lower for prompt in prompts):
                exact_prompt_hits += 1
            if any(cue in lower for cue in instruction_cues):
                instruction_cue_rows += 1
        if row.get("prompt") is not None and row.get("target") is not None:
            prompt_target_rows += 1
            target = row.get("target")
            if isinstance(target, dict):
                structured_target_rows += 1
            elif isinstance(target, str):
                stripped = target.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        if isinstance(json.loads(stripped), dict):
                            structured_target_rows += 1
                    except json.JSONDecodeError:
                        pass

    if total == 0:
        raise ValueError("dataset coverage probe requires at least one record")

    requires_instruction = bool(eval_pack.get("generation_probes"))
    requires_structured = bool(eval_pack.get("structure_tests"))
    requires_length_control = bool(eval_pack.get("length_termination_checks"))
    supervised_instruction_coverage = prompt_target_rows > 0
    supervised_structure_coverage = structured_target_rows > 0
    coverage_gap = (
        (requires_instruction and not supervised_instruction_coverage)
        or (requires_structured and not supervised_structure_coverage)
        or (requires_length_control and not supervised_instruction_coverage)
    )

    measurements: dict[str, object] = {
        "rows_scanned": total,
        "text_rows": text_rows,
        "prompt_target_rows": prompt_target_rows,
        "structured_target_rows": structured_target_rows,
        "exact_frozen_eval_prompt_hits": exact_prompt_hits,
        "instruction_cue_rows": instruction_cue_rows,
        "requires_instruction_behavior": requires_instruction,
        "requires_structured_behavior": requires_structured,
        "requires_length_control": requires_length_control,
        "supervised_instruction_coverage_present": supervised_instruction_coverage,
        "supervised_structure_coverage_present": supervised_structure_coverage,
    }
    evidence = {
        "evidence_kind": "data_audit",
        "source_ref": source_ref,
        "summary": (
            "Scanned the complete processed training dataset for structural coverage of "
            "the frozen semantic_behavior_v1 instruction, structure, and constraint tasks."
        ),
        "confidence": 0.95,
        "data_coverage_gap": coverage_gap,
        "data_coverage_verified": not coverage_gap,
        "measurements": measurements,
    }
    return evidence, measurements


@dataclass(slots=True)
class PostFailureDiagnosticProbe:
    policy: PostFailureProbePolicy = field(default_factory=PostFailureProbePolicy)

    def run(
        self,
        *,
        metrics: Sequence[Mapping[str, object]],
        training_config: Mapping[str, object],
        dataset_rows: Iterable[Mapping[str, object]],
        eval_pack: Mapping[str, object],
        metrics_source_ref: str,
        dataset_source_ref: str,
    ) -> PostFailureProbeResult:
        training_evidence, training_measurements = analyze_training_dynamics(
            metrics,
            training_config,
            source_ref=metrics_source_ref,
            policy=self.policy,
        )
        coverage_evidence, coverage_measurements = analyze_dataset_task_coverage(
            dataset_rows,
            eval_pack,
            source_ref=dataset_source_ref,
        )
        unresolved = [
            "model-family limitation requires a controlled model-family comparison",
            "overfitting requires held-out train/eval-gap evidence",
        ]
        return PostFailureProbeResult(
            evidence_records=[training_evidence, coverage_evidence],
            measurements={
                "training_dynamics": training_measurements,
                "dataset_task_coverage": coverage_measurements,
            },
            unresolved_questions=unresolved,
        )
