"""Evidence-based baseline-versus-candidate behavioral comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from hephaestus.config_loader import ConfigError
from hephaestus.evaluation.pack_loader import load_eval_pack
from hephaestus.evaluators import JudgeModelAdapter
from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.experiment_contract import ExperimentComparison, ExperimentProposal, TrainingRunHandle
from hephaestus.scoring.behavioral import BehavioralSampleScore, evaluate_behavioral_sample
from hephaestus.scoring.comparison import (
    SpreadSummary,
    aggregate_dimensions,
    bounded_repeatability_factor,
    practical_effect,
    summarize_spread,
    variance_risk,
)


_TASK_FIELDS = (
    "generation_probes",
    "continuation_prompts",
    "ranking_sets",
    "regression_prompts",
    "structure_tests",
    "repetition_checks",
    "length_termination_checks",
)
_JUDGE_DIMENSIONS = {"instruction_adherence", "relevance", "coherence"}


@dataclass(slots=True)
class _EvidenceBundle:
    run_id: str
    eval_pack_id: str
    eval_pack_version: str
    integrity_level: str
    content_hash: str | None
    decoding_config: dict[str, object]
    report_ref: str | None
    evidence_refs: list[str]
    samples: list[dict[str, object]]


@dataclass(slots=True)
class _ScoredRun:
    run_id: str
    sample_scores: list[BehavioralSampleScore] = field(default_factory=list)
    dimension_rows: list[dict[str, float]] = field(default_factory=list)
    output_refs: dict[tuple[str, str], str] = field(default_factory=dict)
    missing_samples: list[str] = field(default_factory=list)
    judge_disagreements: list[dict[str, object]] = field(default_factory=list)
    judge_evidence_count: int = 0

    @property
    def hard_failures(self) -> list[str]:
        return sorted(
            {
                f"{score.task_id}:{score.seed}:{name}"
                for score in self.sample_scores
                for name in score.failed_hard_checks
            }
        )

    @property
    def deterministic_pass_rate(self) -> float:
        if not self.sample_scores:
            return 0.0
        return sum(1 for score in self.sample_scores if score.deterministic_passed) / len(self.sample_scores)

    @property
    def overall_scores(self) -> list[float]:
        return [sum(row.values()) / len(row) for row in self.dimension_rows if row]


class ExperimentEvaluationService:
    """Compare recorded behavioral evidence without owning promotion decisions.

    Every run handle must carry a ``semantic_evaluation`` object in ``metadata``.
    Training loss is deliberately ignored because it is not semantic evidence.
    """

    def __init__(
        self,
        *,
        pack_name: str = "semantic_behavior_v1",
        config_dir: Path = Path("configs"),
        judge_adapter: JudgeModelAdapter | None = None,
        minimum_practical_improvement: float | None = None,
    ) -> None:
        self.pack_name = pack_name
        self.pack = load_eval_pack(pack_name, config_dir=config_dir)
        if not bool(self.pack.get("frozen", False)):
            raise ConfigError(f"semantic comparison requires frozen eval pack '{pack_name}'")
        self.judge_adapter = judge_adapter
        scoring = dict(self.pack["eval_pack"].get("scoring_config", {}))
        configured_minimum = float(scoring.get("minimum_practical_improvement", 0.05))
        self.minimum_practical_improvement = (
            configured_minimum
            if minimum_practical_improvement is None
            else max(0.0, float(minimum_practical_improvement))
        )

    def compare(self, proposal: ExperimentProposal, runs: Sequence[TrainingRunHandle]) -> ExperimentComparison:
        comparison_id = f"comparison-{proposal.experiment_id}"
        run_by_id = {run.run_id: run for run in runs}
        baseline_run_id = str(proposal.baseline_ref or "").strip() or None
        issues: list[ContractIssue] = []

        if len(run_by_id) != len(runs):
            issues.append(self._issue("duplicate_run_id", "invalid_request", "Run IDs must be unique.", blocking=True))
        if baseline_run_id is None:
            issues.append(self._issue("baseline_missing", "missing_evidence", "Experiment proposal has no baseline_ref.", blocking=True))
        baseline = run_by_id.get(baseline_run_id or "")
        if baseline_run_id and baseline is None:
            issues.append(self._issue("baseline_run_missing", "missing_evidence", f"Baseline run '{baseline_run_id}' was not provided.", blocking=True))

        candidates = [run for run in runs if run.run_id != baseline_run_id]
        if not candidates:
            issues.append(self._issue("candidate_runs_missing", "missing_evidence", "No candidate runs were provided.", blocking=True))
        for candidate in candidates:
            if candidate.experiment_id != proposal.experiment_id:
                issues.append(
                    self._issue(
                        "candidate_experiment_mismatch",
                        "invalid_request",
                        f"Candidate run '{candidate.run_id}' belongs to experiment '{candidate.experiment_id}'.",
                        blocking=True,
                    )
                )

        candidate_ids = [run.run_id for run in candidates]
        if baseline is None or not candidates or any(issue.blocking for issue in issues):
            return self._invalid_comparison(comparison_id, proposal, baseline_run_id, candidate_ids, issues)

        bundles: dict[str, _EvidenceBundle] = {}
        for run in [baseline, *candidates]:
            bundle, bundle_issues = self._bundle_for(run)
            issues.extend(bundle_issues)
            if bundle is not None:
                bundles[run.run_id] = bundle

        issues.extend(self._validate_settings([baseline, *candidates], bundles))
        if any(issue.blocking for issue in issues):
            return self._invalid_comparison(comparison_id, proposal, baseline_run_id, candidate_ids, issues, bundles)

        tasks = self._tasks()
        seeds = [str(item) for item in dict(self.pack["eval_pack"].get("decoding_config", {})).get("seeds", [])]
        baseline_scored, score_issues = self._score_run(bundles[baseline.run_id], tasks, seeds)
        issues.extend(score_issues)
        candidate_scored: list[_ScoredRun] = []
        for candidate in candidates:
            scored, score_issues = self._score_run(bundles[candidate.run_id], tasks, seeds)
            candidate_scored.append(scored)
            issues.extend(score_issues)

        baseline_dimensions = aggregate_dimensions(baseline_scored.dimension_rows)
        candidate_dimensions = aggregate_dimensions(row for scored in candidate_scored for row in scored.dimension_rows)
        dimension_effects = self._dimension_effects(baseline_dimensions, candidate_dimensions)
        baseline_overall = summarize_spread(baseline_scored.overall_scores)
        candidate_overall = summarize_spread(value for scored in candidate_scored for value in scored.overall_scores)
        overall_delta = candidate_overall.mean - baseline_overall.mean

        scoring_config = dict(self.pack["eval_pack"].get("scoring_config", {}))
        candidate_repeat_spread = summarize_spread(
            value
            for scored in candidate_scored
            for value in self._repeat_unit_scores(scored)
        )
        observed_variance = variance_risk(
            candidate_repeat_spread,
            float(scoring_config.get("moderate_variance_threshold", 0.06)),
            float(scoring_config.get("high_variance_threshold", 0.12)),
        )
        consistency = self._effect_consistency(baseline_scored, candidate_scored, overall_delta)
        minimum_consistency = float(scoring_config.get("minimum_direction_consistency", 0.67))
        candidate_hard_failures = sorted({failure for scored in candidate_scored for failure in scored.hard_failures})
        missing_samples = sorted(
            set(baseline_scored.missing_samples).union(*(set(scored.missing_samples) for scored in candidate_scored))
        )
        candidate_disagreements = [item for scored in candidate_scored for item in scored.judge_disagreements]
        disagreements = [*baseline_scored.judge_disagreements, *candidate_disagreements]
        outcome = self._outcome(
            overall_delta,
            dimension_effects,
            candidate_hard_failures,
            missing_samples,
            observed_variance,
            consistency,
            minimum_consistency,
            candidate_disagreements,
        )

        expected_per_run = max(1, len(tasks) * max(1, len(seeds)))
        observed_per_run = [len(baseline_scored.sample_scores), *(len(item.sample_scores) for item in candidate_scored)]
        completeness = sum(min(count / expected_per_run, 1.0) for count in observed_per_run) / len(observed_per_run)
        required_runs = int(dict(self.pack["eval_pack"].get("required_evidence", {})).get("comparison_runs", 2))
        repeatability_factor = bounded_repeatability_factor(len(candidate_scored), required_runs)
        confidence, ceiling = self._confidence(
            completeness,
            repeatability_factor,
            consistency,
            observed_variance,
            missing_samples,
            disagreements,
            list(bundles.values()),
        )
        deterministic_status = "failed" if candidate_hard_failures else "incomplete" if missing_samples else "passed"
        report_refs = self._unique(bundle.report_ref for bundle in bundles.values())
        evidence_refs = self._unique(ref for bundle in bundles.values() for ref in bundle.evidence_refs)
        human_review = self._human_review_bundle(comparison_id, tasks, baseline_scored, candidate_scored, disagreements)

        return ExperimentComparison(
            comparison_id=comparison_id,
            experiment_id=proposal.experiment_id,
            baseline_run_id=baseline_run_id,
            candidate_run_ids=candidate_ids,
            evaluation_report_refs=report_refs,
            primary_outcome=outcome,
            effect_summary={
                "minimum_practical_improvement": self.minimum_practical_improvement,
                "baseline": baseline_overall.to_dict(),
                "candidate": candidate_overall.to_dict(),
                "overall_delta": overall_delta,
                "overall_effect": practical_effect(overall_delta, self.minimum_practical_improvement),
                "dimensions": dimension_effects,
                "deterministic": {
                    "baseline_pass_rate": baseline_scored.deterministic_pass_rate,
                    "candidate_pass_rate": sum(item.deterministic_pass_rate for item in candidate_scored) / len(candidate_scored),
                    "candidate_hard_failures": candidate_hard_failures,
                },
                "repeatability": {
                    "candidate_run_count": len(candidate_scored),
                    "candidate_sample_count": candidate_overall.count,
                    "direction_consistency": consistency,
                    "variance_risk": observed_variance,
                    "repeat_unit_spread": candidate_repeat_spread.to_dict(),
                    "formal_significance_claimed": False,
                },
                "judge": {
                    "adapter_used": self.judge_adapter is not None,
                    "evidence_count": sum(item.judge_evidence_count for item in [baseline_scored, *candidate_scored]),
                    "disagreements": disagreements,
                    "deterministic_precedence": True,
                },
                "missing_evidence": missing_samples,
            },
            deterministic_gate_status=deterministic_status,
            variance_risk=observed_variance,
            recommendation=self._recommendation(outcome),
            evidence_refs=evidence_refs,
            issues=issues,
            confidence=confidence,
            metadata={
                "eval_pack_id": self.pack["eval_pack_id"],
                "eval_pack_version": self.pack["eval_pack_version"],
                "eval_pack_content_hash": self.pack.get("content_hash"),
                "comparison_configuration": {
                    "decoding_config": dict(self.pack["eval_pack"].get("decoding_config", {})),
                    "minimum_practical_improvement": self.minimum_practical_improvement,
                    "minimum_direction_consistency": minimum_consistency,
                },
                "confidence_ceiling": ceiling,
                "human_review_bundle": human_review,
                "does_not_promote": True,
            },
        )

    def _bundle_for(self, run: TrainingRunHandle) -> tuple[_EvidenceBundle | None, list[ContractIssue]]:
        raw = run.metadata.get("semantic_evaluation")
        if not isinstance(raw, dict):
            return None, [self._issue("semantic_evaluation_missing", "missing_evidence", f"Run '{run.run_id}' has no semantic_evaluation artifact metadata.", blocking=True)]
        issues: list[ContractIssue] = []
        samples = raw.get("samples")
        if not isinstance(samples, list):
            issues.append(self._issue("evaluation_samples_missing", "missing_evidence", f"Run '{run.run_id}' has no sample list.", blocking=True))
            samples = []
        report_ref = str(raw.get("report_ref") or "").strip() or None
        if report_ref is None:
            issues.append(self._issue("evaluation_report_ref_missing", "missing_evidence", f"Run '{run.run_id}' has no evaluation report reference."))
        sample_refs = [
            str(item.get("evidence_ref"))
            for item in samples
            if isinstance(item, dict) and item.get("evidence_ref")
        ]
        return _EvidenceBundle(
            run_id=run.run_id,
            eval_pack_id=str(raw.get("eval_pack_id") or ""),
            eval_pack_version=str(raw.get("eval_pack_version") or ""),
            integrity_level=str(raw.get("integrity_level") or "insufficient"),
            content_hash=str(raw.get("content_hash") or "").strip() or None,
            decoding_config=dict(raw.get("decoding_config", {})) if isinstance(raw.get("decoding_config", {}), dict) else {},
            report_ref=report_ref,
            evidence_refs=self._unique([*[str(item) for item in raw.get("evidence_refs", [])], *sample_refs]),
            samples=[dict(item) for item in samples if isinstance(item, dict)],
        ), issues

    def _validate_settings(self, runs: list[TrainingRunHandle], bundles: dict[str, _EvidenceBundle]) -> list[ContractIssue]:
        issues: list[ContractIssue] = []
        expected_id = str(self.pack["eval_pack_id"])
        expected_version = str(self.pack["eval_pack_version"])
        expected_hash = str(self.pack.get("content_hash") or "") or None
        expected_decoding = dict(self.pack["eval_pack"].get("decoding_config", {}))
        for run in runs:
            bundle = bundles.get(run.run_id)
            if bundle is None:
                continue
            if not bundle.eval_pack_id or not bundle.eval_pack_version:
                issues.append(self._issue("eval_pack_identity_missing", "artifact_integrity", f"Run '{run.run_id}' is missing eval-pack identity or version.", blocking=True))
            elif (bundle.eval_pack_id, bundle.eval_pack_version) != (expected_id, expected_version):
                issues.append(self._issue("eval_pack_mismatch", "artifact_integrity", f"Run '{run.run_id}' used {bundle.eval_pack_id}@{bundle.eval_pack_version}; expected {expected_id}@{expected_version}.", blocking=True))
            allowed_integrity = {"content_hash_verified", "reference_only", "inline_unhashed"}
            if bundle.integrity_level not in allowed_integrity:
                issues.append(self._issue("eval_pack_integrity_missing", "artifact_integrity", f"Run '{run.run_id}' lacks usable eval-pack integrity evidence.", blocking=True))
            if expected_hash and (
                bundle.content_hash != expected_hash
                or bundle.integrity_level != "content_hash_verified"
            ):
                issues.append(self._issue("eval_pack_content_hash_mismatch", "artifact_integrity", f"Run '{run.run_id}' has missing or mismatched eval-pack content hash.", blocking=True))
            if self._canonical(bundle.decoding_config) != self._canonical(expected_decoding):
                issues.append(
                    self._issue(
                        "decoding_settings_mismatch",
                        "invalid_request",
                        f"Run '{run.run_id}' decoding settings are incompatible with the frozen pack.",
                        blocking=True,
                        metadata={"observed": bundle.decoding_config, "expected": expected_decoding},
                    )
                )
        return issues

    def _score_run(
        self,
        bundle: _EvidenceBundle,
        tasks: dict[str, dict[str, object]],
        seeds: list[str],
    ) -> tuple[_ScoredRun, list[ContractIssue]]:
        scored = _ScoredRun(run_id=bundle.run_id)
        issues: list[ContractIssue] = []
        observed: dict[tuple[str, str], dict[str, object]] = {}
        for sample in bundle.samples:
            task_id = str(sample.get("task_id") or "")
            seed = str(sample.get("seed") if sample.get("seed") is not None else "")
            key = (task_id, seed)
            if not task_id or not seed:
                issues.append(self._issue("sample_identity_missing", "missing_evidence", f"Run '{bundle.run_id}' has a sample without task_id or seed."))
            elif key in observed:
                issues.append(self._issue("duplicate_sample", "invalid_request", f"Run '{bundle.run_id}' has duplicate sample {task_id}/{seed}."))
            else:
                observed[key] = sample

        expected = {(task_id, seed) for task_id in tasks for seed in seeds} if seeds else set(observed)
        scored.missing_samples = [f"{bundle.run_id}:{task_id}:{seed}" for task_id, seed in sorted(expected - set(observed))]
        if scored.missing_samples:
            issues.append(
                self._issue(
                    "required_samples_missing",
                    "missing_evidence",
                    f"Run '{bundle.run_id}' is missing {len(scored.missing_samples)} required samples.",
                    retryable=True,
                    metadata={"missing_samples": list(scored.missing_samples)},
                )
            )

        scoring_config = dict(self.pack["eval_pack"].get("scoring_config", {}))
        favorable = float(scoring_config.get("judge_favorable_threshold", 0.75))
        unfavorable = float(scoring_config.get("judge_unfavorable_threshold", 0.4))
        for (task_id, seed), sample in sorted(observed.items()):
            task = tasks.get(task_id)
            if task is None:
                issues.append(self._issue("unknown_eval_task", "invalid_request", f"Run '{bundle.run_id}' contains unknown task '{task_id}'."))
                continue
            response = str(sample.get("output") or "")
            evidence_ref = str(sample.get("evidence_ref") or "").strip()
            if not evidence_ref:
                missing_ref = f"{bundle.run_id}:{task_id}:{seed}:evidence_ref"
                scored.missing_samples.append(missing_ref)
                issues.append(
                    self._issue(
                        "sample_evidence_ref_missing",
                        "missing_evidence",
                        f"Run '{bundle.run_id}' sample {task_id}/{seed} has no evidence reference.",
                    )
                )
            sample_score = evaluate_behavioral_sample(task, response, seed)
            combined = dict(sample_score.dimension_scores)
            judge_scores: dict[str, float] = {}
            if self.judge_adapter is not None:
                try:
                    judge_scores = {
                        str(name): max(0.0, min(1.0, float(value)))
                        for name, value in self.judge_adapter.score(task, response).items()
                        if str(name) in _JUDGE_DIMENSIONS
                    }
                except Exception as exc:
                    issues.append(self._issue("judge_adapter_failed", "provider_unavailable", f"Judge adapter failed for {bundle.run_id}/{task_id}/{seed}: {exc.__class__.__name__}.", retryable=True))
            if judge_scores:
                scored.judge_evidence_count += 1
                for dimension, judge_score in judge_scores.items():
                    combined[dimension] = (combined[dimension] + judge_score) / 2.0 if dimension in combined else judge_score
                judge_mean = sum(judge_scores.values()) / len(judge_scores)
                if not sample_score.deterministic_passed and judge_mean >= favorable:
                    scored.judge_disagreements.append({"run_id": bundle.run_id, "task_id": task_id, "seed": seed, "kind": "judge_favorable_deterministic_failed", "judge_mean": judge_mean, "failed_hard_checks": sample_score.failed_hard_checks})
                elif sample_score.deterministic_passed and judge_mean < unfavorable:
                    scored.judge_disagreements.append({"run_id": bundle.run_id, "task_id": task_id, "seed": seed, "kind": "deterministic_pass_judge_unfavorable", "judge_mean": judge_mean, "failed_hard_checks": []})
            scored.sample_scores.append(sample_score)
            scored.dimension_rows.append(combined)
            scored.output_refs[(task_id, seed)] = evidence_ref
        return scored, issues

    def _tasks(self) -> dict[str, dict[str, object]]:
        normalized = dict(self.pack["eval_pack"])
        tasks: dict[str, dict[str, object]] = {}
        for field_name in _TASK_FIELDS:
            raw_tasks = normalized.get(field_name, [])
            if not isinstance(raw_tasks, list):
                continue
            for raw in raw_tasks:
                if not isinstance(raw, dict):
                    continue
                task = dict(raw)
                task_id = str(task.get("task_id") or "")
                if not task_id:
                    continue
                task.setdefault("task_kind", field_name)
                if task_id in tasks:
                    raise ValueError(f"duplicate semantic evaluation task_id: {task_id}")
                tasks[task_id] = task
        if not tasks:
            raise ValueError(f"eval pack '{self.pack_name}' contains no semantic tasks")
        return tasks

    def _dimension_effects(self, baseline: dict[str, SpreadSummary], candidate: dict[str, SpreadSummary]) -> dict[str, dict[str, object]]:
        effects: dict[str, dict[str, object]] = {}
        for dimension in sorted(set(baseline).intersection(candidate)):
            delta = candidate[dimension].mean - baseline[dimension].mean
            effects[dimension] = {
                "baseline_mean": baseline[dimension].mean,
                "candidate_mean": candidate[dimension].mean,
                "delta": delta,
                "effect": practical_effect(delta, self.minimum_practical_improvement),
                "baseline_count": baseline[dimension].count,
                "candidate_count": candidate[dimension].count,
            }
        return effects

    def _effect_consistency(self, baseline: _ScoredRun, candidates: list[_ScoredRun], overall_delta: float) -> float:
        baseline_values = {(score.task_id, str(score.seed)): score.overall_score for score in baseline.sample_scores}
        deltas = [
            score.overall_score - baseline_values[(score.task_id, str(score.seed))]
            for candidate in candidates
            for score in candidate.sample_scores
            if (score.task_id, str(score.seed)) in baseline_values
        ]
        if not deltas:
            return 0.0
        threshold = self.minimum_practical_improvement
        if overall_delta >= threshold:
            return sum(1 for delta in deltas if delta >= 0.0) / len(deltas)
        if overall_delta <= -threshold:
            return sum(1 for delta in deltas if delta <= 0.0) / len(deltas)
        return sum(1 for delta in deltas if abs(delta) < threshold) / len(deltas)

    def _repeat_unit_scores(self, scored: _ScoredRun) -> list[float]:
        by_seed: dict[str, list[float]] = {}
        for sample, dimensions in zip(scored.sample_scores, scored.dimension_rows, strict=True):
            if dimensions:
                by_seed.setdefault(str(sample.seed), []).append(sum(dimensions.values()) / len(dimensions))
        return [sum(values) / len(values) for _, values in sorted(by_seed.items()) if values]

    def _outcome(
        self,
        overall_delta: float,
        dimension_effects: dict[str, dict[str, object]],
        hard_failures: list[str],
        missing_samples: list[str],
        observed_variance: str,
        consistency: float,
        minimum_consistency: float,
        disagreements: list[dict[str, object]],
    ) -> str:
        if hard_failures:
            return "regressed"
        if missing_samples or not dimension_effects:
            return "inconclusive"
        effects = {str(item["effect"]) for item in dimension_effects.values()}
        if "improved" in effects and "regressed" in effects:
            return "mixed"
        if disagreements and ("improved" in effects or "regressed" in effects):
            return "mixed"
        overall = practical_effect(overall_delta, self.minimum_practical_improvement)
        if overall == "improved":
            return "improved" if observed_variance == "low" and consistency >= minimum_consistency else "inconclusive"
        if overall == "regressed":
            return "regressed"
        return "equivalent_within_evidence"

    def _confidence(
        self,
        completeness: float,
        repeatability_factor: float,
        consistency: float,
        variance: str,
        missing_samples: list[str],
        disagreements: list[dict[str, object]],
        bundles: list[_EvidenceBundle],
    ) -> tuple[float, float]:
        ceiling = 0.9
        if self.judge_adapter is None:
            ceiling = min(ceiling, 0.75)
        if any(bundle.integrity_level != "content_hash_verified" for bundle in bundles):
            ceiling = min(ceiling, 0.55)
        if missing_samples:
            ceiling = min(ceiling, 0.45)
        if variance == "high":
            ceiling = min(ceiling, 0.5)
        elif variance in {"moderate", "unknown"}:
            ceiling = min(ceiling, 0.7)
        if disagreements:
            ceiling = min(ceiling, 0.6)
        raw = completeness * repeatability_factor * max(0.25, consistency)
        return max(0.0, min(raw, ceiling)), ceiling

    def _human_review_bundle(
        self,
        comparison_id: str,
        tasks: dict[str, dict[str, object]],
        baseline: _ScoredRun,
        candidates: list[_ScoredRun],
        disagreements: list[dict[str, object]],
    ) -> dict[str, object]:
        rows = [
            {
                "task_id": task_id,
                "seed": seed,
                "prompt": str(tasks[task_id].get("prompt", "")),
                "baseline_run_id": baseline.run_id,
                "baseline_output_ref": baseline.output_refs[(task_id, seed)],
                "candidate_run_id": candidate.run_id,
                "candidate_output_ref": candidate.output_refs[(task_id, seed)],
            }
            for candidate in candidates
            for task_id, seed in sorted(set(baseline.output_refs).intersection(candidate.output_refs))
        ]
        return {
            "bundle_id": f"human-review-{comparison_id}",
            "persistence_status": "inline_bundle_requires_caller_persistence",
            "blind_review_recommended": True,
            "rubric": ["instruction_adherence", "relevance", "coherence", "factual_or_task_correctness"],
            "samples": rows,
            "judge_disagreements": disagreements,
        }

    def _invalid_comparison(
        self,
        comparison_id: str,
        proposal: ExperimentProposal,
        baseline_run_id: str | None,
        candidate_ids: list[str],
        issues: list[ContractIssue],
        bundles: dict[str, _EvidenceBundle] | None = None,
    ) -> ExperimentComparison:
        bundle_values = list((bundles or {}).values())
        return ExperimentComparison(
            comparison_id=comparison_id,
            experiment_id=proposal.experiment_id,
            baseline_run_id=baseline_run_id,
            candidate_run_ids=candidate_ids,
            evaluation_report_refs=self._unique(bundle.report_ref for bundle in bundle_values),
            primary_outcome="invalid_comparison",
            effect_summary={"reason": "incompatible_or_missing_comparison_evidence"},
            deterministic_gate_status="incompatible",
            variance_risk="unknown",
            recommendation="repair_evaluation_evidence",
            evidence_refs=self._unique(ref for bundle in bundle_values for ref in bundle.evidence_refs),
            issues=issues,
            confidence=0.0,
            metadata={"eval_pack_id": self.pack.get("eval_pack_id"), "eval_pack_version": self.pack.get("eval_pack_version"), "confidence_ceiling": 0.0, "does_not_promote": True},
        )

    def _recommendation(self, outcome: str) -> str:
        return {
            "improved": "consider_candidate_after_human_review",
            "regressed": "reject_candidate_evidence",
            "equivalent_within_evidence": "retain_baseline",
            "mixed": "human_review_required",
            "inconclusive": "collect_more_evidence",
            "invalid_comparison": "repair_evaluation_evidence",
        }.get(outcome, "collect_more_evidence")

    def _issue(
        self,
        code: str,
        category: str,
        message: str,
        *,
        retryable: bool = False,
        blocking: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> ContractIssue:
        return ContractIssue(code, category, message, retryable=retryable, blocking=blocking, metadata=metadata or {})

    def _canonical(self, payload: Mapping[str, object]) -> str:
        return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def _unique(self, values: Iterable[str | None]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result
