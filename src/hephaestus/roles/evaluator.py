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
        stage_thresholds = dict(eval_pack["stage_tolerances"].get(stage_profile.strictness, {}))
        cert_required_metrics = set(eval_pack["certification_bundle"]["required_metrics"])
        cert_metric_ready = cert_required_metrics.issubset(required_metrics)

        stage_cert = dict(stage_profile.certification_profile)
        stage_eligibility = str(stage_cert.get("eligibility", "standard"))
        stage_require_recheck = bool(stage_cert.get("require_recheck", False))
        stage_min_consistent = int(stage_cert.get("min_consistent_runs", 1))
        effective_min_consistent = max(int(recheck_rules["min_consistent_runs"]), stage_min_consistent)
        consistent_runs = self._count_consistent_runs(certification_evals)
        consistency_passed = consistent_runs >= effective_min_consistent
        recheck_required = bool(recheck_rules["required_for_certification"] or stage_require_recheck)
        recheck_recommended = bool(recheck_required and not consistency_passed)

        evidence_total = max(int(evidence_rules["stable_runs"]), 1)
        observed_evidence = 1 + len(certification_evals)
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
        elif recheck_recommended:
            certification_readiness = "certification_inconclusive"
        elif not cert_metric_ready:
            certification_readiness = "certification_not_eligible"
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
                },
                "observed_consistent_runs": consistent_runs,
                "min_stability_confidence": float(eval_pack["certification_bundle"]["min_stability_confidence"]),
                "stage_thresholds": stage_thresholds,
            },
            intermediate_artifact_refs=[ref for ref in refs if ref],
        )

    def _count_consistent_runs(self, certification_evals: list[object]) -> int:
        consistent = 0
        for item in certification_evals:
            if not isinstance(item, dict):
                continue
            if bool(item.get("deterministic_passed", False)):
                consistent += 1
        return consistent
