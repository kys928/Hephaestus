from __future__ import annotations

from dataclasses import dataclass

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.evaluation.checkpoint_selector import select_checkpoint
from hephaestus.evaluation.metric_reader import read_metrics
from hephaestus.evaluation.pack_loader import load_eval_pack
from hephaestus.evaluation.regression_checks import build_regression_summary
from hephaestus.evaluation.stage_interpreter import interpret_stage
from hephaestus.schemas.checkpoint_resolution import CheckpointResolution
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.metric_summary import MetricSummary
from hephaestus.schemas.stage_profile import StageProfile


@dataclass(slots=True)
class EvaluatorRole:
    backend: DryRunBackend
    name: str = "evaluator"

    def run(self, run_id: str, stage_profile: StageProfile) -> EvalReport:
        launch_result = self.backend.launch_training(run_id)
        eval_pack = load_eval_pack(stage_profile.eval_pack)
        intermediate = launch_result["intermediate_eval"]
        metrics = read_metrics(intermediate)
        regression = build_regression_summary(
            run_id=run_id,
            probe_score=metrics["probe_score"],
            toxicity=metrics["toxicity"],
            min_probe_score=stage_profile.deterministic_gates["min_probe_score"],
            max_toxicity=stage_profile.deterministic_gates["max_toxicity"],
        )
        confidence, issue = interpret_stage(stage_profile.strictness, regression.deterministic_passed)
        metric_summaries = [
            MetricSummary("probe_score", metrics["probe_score"], stage_profile.deterministic_gates["min_probe_score"], metrics["probe_score"] >= stage_profile.deterministic_gates["min_probe_score"]).to_dict(),
            MetricSummary("toxicity", metrics["toxicity"], stage_profile.deterministic_gates["max_toxicity"], metrics["toxicity"] <= stage_profile.deterministic_gates["max_toxicity"]).to_dict(),
        ]
        checkpoint = select_checkpoint([dict(candidate) for candidate in launch_result["checkpoint_candidates"]])
        checkpoint_resolution = CheckpointResolution(
            run_id=run_id,
            selected_checkpoint_ref=str(checkpoint["checkpoint_ref"]),
            reason="best_probe_score",
            confidence=confidence,
        )
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
            intermediate_artifact_refs=[
                str(intermediate["probe_ref"]),
                str(intermediate["metrics_ref"]),
                str(intermediate["deterministic_ref"]),
            ],
        )
