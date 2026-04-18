from __future__ import annotations

from dataclasses import dataclass

from hephaestus.config_loader import ConfigError
from hephaestus.evaluation.checkpoint_selector import select_checkpoint
from hephaestus.evaluation.metric_reader import MetricsArtifactError, read_metrics
from hephaestus.evaluation.pack_loader import load_eval_pack
from hephaestus.evaluation.regression_checks import build_regression_summary
from hephaestus.evaluation.stage_interpreter import interpret_stage
from hephaestus.schemas.checkpoint_resolution import CheckpointResolution
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.metric_summary import MetricSummary
from hephaestus.schemas.stage_profile import StageProfile


_SUPPORTED_METRICS = {"probe_score", "toxicity"}
_VARIANCE_THRESHOLDS = {"low": 0.14, "medium": 0.08, "high": 0.04}
_VARIANCE_LEVEL = {"low": 0, "medium": 1, "high": 2}


@dataclass(slots=True)
class EvaluatorRole:
    name: str = "evaluator"

    def run(self, run_id: str, stage_profile: StageProfile, training_outputs: dict[str, object]) -> EvalReport:
        eval_pack = load_eval_pack(stage_profile.eval_pack)
        required_metrics = set(eval_pack["required_metrics"])
        unsupported = sorted(required_metrics - _SUPPORTED_METRICS)
        if unsupported:
            raise ConfigError(f"eval pack '{stage_profile.eval_pack}' uses unsupported metrics: {', '.join(unsupported)}")

        intermediate = dict(training_outputs.get("intermediate_eval", {}))
        certification_evals = training_outputs.get("certification_evals", [])
        if not isinstance(certification_evals, list):
            certification_evals = []

        try:
            metrics = read_metrics(intermediate)
            metrics_missing = False
        except MetricsArtifactError:
            metrics = {"probe_score": 0.0, "toxicity": 1.0}
            metrics_missing = True

        regression = build_regression_summary(
            run_id=run_id,
            metrics=metrics,
            gates=stage_profile.deterministic_gates,
            regression_bundles=dict(eval_pack["regression_bundles"]),
        )
        confidence, issue = interpret_stage(stage_profile.strictness, regression.deterministic_passed)

        promotion_bundle = regression.bundle_results.get("promotion", {"passed": regression.deterministic_passed})
        certification_bundle_name = str(eval_pack["certification_bundle"]["required_bundle"])
        certification_bundle = regression.bundle_results.get(certification_bundle_name, {"passed": False})

        evidence_rules = dict(eval_pack["minimum_evidence"])
        recheck_rules = dict(eval_pack["recheck_requirements"])
        repeatability_rules = dict(eval_pack.get("repeatability_requirements", {}))
        stage_thresholds = dict(eval_pack["stage_tolerances"].get(stage_profile.strictness, {}))
        cert_required_metrics = set(eval_pack["certification_bundle"]["required_metrics"])
        cert_metric_ready = cert_required_metrics.issubset(required_metrics)

        stage_cert = dict(stage_profile.certification_profile)
        stage_eligibility = str(stage_cert.get("eligibility", "standard"))
        stage_require_recheck = bool(stage_cert.get("require_recheck", False))
        stage_min_consistent = int(stage_cert.get("min_consistent_runs", 1))

        repeated_eval_count = len(certification_evals)
        observed_evidence = 1 + repeated_eval_count
        consistent_runs = self._count_consistent_runs(certification_evals, primary_deterministic_passed=regression.deterministic_passed)
        consistency_score = consistent_runs / observed_evidence if observed_evidence > 0 else 0.0

        effective_min_consistent = max(int(recheck_rules["min_consistent_runs"]), stage_min_consistent)
        consistency_passed = consistent_runs >= effective_min_consistent

        effective_repeatability_required = bool(
            repeatability_rules.get("repeatability_required", False) or stage_cert.get("repeatability_required", False)
        )
        effective_required_rechecks = max(
            int(repeatability_rules.get("required_rechecks", 0)),
            int(stage_cert.get("required_rechecks", 0)),
        )
        effective_min_repeat_consistency = max(
            float(repeatability_rules.get("min_repeat_consistency", 0.0)),
            float(stage_cert.get("min_repeat_consistency", 0.0)),
        )
        variance_sensitivity = self._resolve_variance_sensitivity(
            str(repeatability_rules.get("variance_sensitivity", "medium")),
            str(stage_cert.get("variance_sensitivity", "medium")),
        )
        recheck_policy = self._resolve_recheck_policy(
            str(repeatability_rules.get("certification_recheck_policy", "required_if_repeatability_unmet")),
            str(stage_cert.get("certification_recheck_policy", "required_if_repeatability_unmet")),
        )

        variance_risk = self._variance_risk(metrics, certification_evals, variance_sensitivity)
        repeatability_sufficient = (
            (not effective_repeatability_required)
            or (
                repeated_eval_count >= effective_required_rechecks
                and consistency_score >= effective_min_repeat_consistency
                and variance_risk != "high"
            )
        )
        repeatability_blocked = bool(effective_repeatability_required and not repeatability_sufficient)

        recheck_required = bool(recheck_rules["required_for_certification"] or stage_require_recheck)
        recheck_needed = self._recheck_needed(
            recheck_required=recheck_required,
            recheck_policy=recheck_policy,
            repeatability_blocked=repeatability_blocked,
            variance_risk=variance_risk,
        )
        recheck_recommended = bool(recheck_needed or (recheck_required and not consistency_passed))

        evidence_total = max(int(evidence_rules["stable_runs"]), 1)
        evidence_completeness = min(observed_evidence / evidence_total, 1.0)
        stability_confidence = min(confidence * evidence_completeness, 1.0)

        if metrics_missing:
            confidence = 0.0
            stability_confidence = 0.0
            issue = "metrics_artifact_missing"
            certification_readiness = "certification_not_eligible"
        elif stage_eligibility in {"disabled", "ineligible", "none"}:
            certification_readiness = "certification_not_eligible"
        elif not bool(certification_bundle.get("passed", False)):
            certification_readiness = "certification_blocked_by_regression"
        elif evidence_completeness < 1.0:
            certification_readiness = "certification_inconclusive"
        elif not cert_metric_ready:
            certification_readiness = "certification_not_eligible"
        elif recheck_needed and repeated_eval_count < effective_required_rechecks:
            certification_readiness = "certification_recheck_required"
        elif variance_risk == "high" and effective_repeatability_required:
            certification_readiness = "certification_inconclusive_due_to_variance"
        elif consistency_score < effective_min_repeat_consistency and effective_repeatability_required:
            certification_readiness = "certification_blocked_by_inconsistency"
        elif recheck_recommended:
            certification_readiness = "certification_recheck_required"
        else:
            certification_readiness = "certification_passed"

        metric_summaries = [
            MetricSummary(
                "probe_score",
                metrics["probe_score"],
                stage_profile.deterministic_gates["min_probe_score"],
                metrics["probe_score"] >= stage_profile.deterministic_gates["min_probe_score"],
            ).to_dict(),
            MetricSummary(
                "toxicity",
                metrics["toxicity"],
                stage_profile.deterministic_gates["max_toxicity"],
                metrics["toxicity"] <= stage_profile.deterministic_gates["max_toxicity"],
            ).to_dict(),
        ]
        checkpoint = select_checkpoint([dict(candidate) for candidate in training_outputs.get("checkpoint_candidates", [])])
        checkpoint_reason = str(checkpoint.get("reason", "best_probe_score"))
        if metrics_missing:
            checkpoint_reason = "metrics_artifact_missing"
        checkpoint_resolution = CheckpointResolution(
            run_id=run_id,
            selected_checkpoint_ref=str(checkpoint.get("checkpoint_ref", "")),
            reason=checkpoint_reason,
            confidence=confidence,
        )
        refs = [
            str(intermediate.get("probe_ref", "")),
            str(intermediate.get("metrics_ref", "")),
            str(intermediate.get("deterministic_ref", "")),
        ]
        return EvalReport(
            eval_id=f"eval-{run_id}",
            run_id=run_id,
            stage_name=stage_profile.name,
            pack_name=str(eval_pack["pack_name"]),
            metric_summaries=metric_summaries,
            regression_summary=regression.to_dict(),
            checkpoint_resolution=checkpoint_resolution.to_dict(),
            confidence=confidence,
            likely_issue_category=issue,
            evidence_completeness=evidence_completeness,
            stability_confidence=stability_confidence,
            certification_readiness=certification_readiness,
            recheck_recommended=recheck_recommended,
            promotion_bundle_passed=bool(promotion_bundle.get("passed", False)),
            observed_consistent_runs=consistent_runs,
            repeated_eval_count=repeated_eval_count,
            consistency_score=consistency_score,
            repeatability_ready=repeatability_sufficient,
            repeatability_blocked=repeatability_blocked,
            repeatability_sufficient=repeatability_sufficient,
            recheck_needed=recheck_needed,
            variance_risk=variance_risk,
            consistency_observed=self._consistency_observed(consistency_score, effective_min_repeat_consistency),
            certification_recheck_count=effective_required_rechecks,
            evaluation_bundle_summary={
                "required_metrics": list(eval_pack["required_metrics"]),
                "certification_required_metrics": list(cert_required_metrics),
                "promotion_bundle_passed": bool(promotion_bundle.get("passed", False)),
                "certification_bundle": certification_bundle_name,
                "certification_bundle_passed": bool(certification_bundle.get("passed", False)),
                "observed_evidence_runs": observed_evidence,
                "required_evidence": evidence_rules,
                "consistency_passed": consistency_passed,
                "effective_min_consistent_runs": effective_min_consistent,
                "stage_certification_profile": {
                    "eligibility": stage_eligibility,
                    "require_recheck": stage_require_recheck,
                    "min_consistent_runs": stage_min_consistent,
                    "repeatability_required": bool(stage_cert.get("repeatability_required", False)),
                    "required_rechecks": int(stage_cert.get("required_rechecks", 0)),
                    "min_repeat_consistency": float(stage_cert.get("min_repeat_consistency", 0.0)),
                    "variance_sensitivity": str(stage_cert.get("variance_sensitivity", "medium")),
                    "certification_recheck_policy": str(
                        stage_cert.get("certification_recheck_policy", "required_if_repeatability_unmet")
                    ),
                },
                "observed_consistent_runs": consistent_runs,
                "min_stability_confidence": float(eval_pack["certification_bundle"]["min_stability_confidence"]),
                "stage_thresholds": stage_thresholds,
                "repeatability": {
                    "repeatability_required": effective_repeatability_required,
                    "required_rechecks": effective_required_rechecks,
                    "min_repeat_consistency": effective_min_repeat_consistency,
                    "variance_sensitivity": variance_sensitivity,
                    "variance_risk": variance_risk,
                    "recheck_policy": recheck_policy,
                    "repeatability_sufficient": repeatability_sufficient,
                },
            },
            intermediate_artifact_refs=[ref for ref in refs if ref],
        )

    def _count_consistent_runs(self, certification_evals: list[object], primary_deterministic_passed: bool) -> int:
        consistent = 1 if primary_deterministic_passed else 0
        for item in certification_evals:
            if not isinstance(item, dict):
                continue
            if bool(item.get("deterministic_passed", False)):
                consistent += 1
        return consistent

    def _resolve_variance_sensitivity(self, pack_value: str, stage_value: str) -> str:
        pack = pack_value if pack_value in _VARIANCE_LEVEL else "medium"
        stage = stage_value if stage_value in _VARIANCE_LEVEL else "medium"
        return stage if stage_value in _VARIANCE_LEVEL else pack

    def _resolve_recheck_policy(self, pack_value: str, stage_value: str) -> str:
        allowed = {"always", "never", "required_if_repeatability_unmet", "required_if_variance"}
        if stage_value in allowed:
            return stage_value
        if pack_value in allowed:
            return pack_value
        return "required_if_repeatability_unmet"

    def _variance_risk(self, metrics: dict[str, float], certification_evals: list[object], variance_sensitivity: str) -> str:
        probe_scores = [float(metrics.get("probe_score", 0.0))]
        for item in certification_evals:
            if not isinstance(item, dict):
                continue
            if "probe_score" in item:
                try:
                    probe_scores.append(float(item["probe_score"]))
                except (TypeError, ValueError):
                    continue
        if len(probe_scores) < 2:
            return "unknown"
        spread = max(probe_scores) - min(probe_scores)
        threshold = _VARIANCE_THRESHOLDS.get(variance_sensitivity, _VARIANCE_THRESHOLDS["medium"])
        if spread > threshold:
            return "high"
        if spread > threshold * 0.6:
            return "moderate"
        return "low"

    def _recheck_needed(self, recheck_required: bool, recheck_policy: str, repeatability_blocked: bool, variance_risk: str) -> bool:
        if recheck_policy == "never":
            return False
        if recheck_policy == "always":
            return True
        if recheck_policy == "required_if_variance":
            return variance_risk == "high"
        return repeatability_blocked

    def _consistency_observed(self, score: float, minimum: float) -> str:
        if minimum <= 0.0 and score <= 0.0:
            return "unknown"
        if score >= minimum:
            return "consistent"
        if score >= max(0.0, minimum - 0.15):
            return "mixed"
        return "inconsistent"
