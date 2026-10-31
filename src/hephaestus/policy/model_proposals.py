"""Explicit translations; trained actions are proposals, never execution grants."""
from hephaestus.schemas.model_proposal import ModelInterventionSelection

# decision -> (domain intervention, required trained action, model -> domain variable)
PLANNER_DECISIONS = {
    "collect_more_evidence": ("collect_more_evidence", "request_recheck", {
        "diagnostic_measurement":"diagnostic_measurement", "experiment_budget":"diagnostic_measurement"}),
    "repair_evaluation": ("repair_evaluation", "request_recheck", {"evaluation_protocol":"evaluation_protocol"}),
    "change_preprocessing": ("change_preprocessing", "branch_new_experiment", {"preprocessing_policy":"preprocessing_policy"}),
    "change_model": ("change_model", "branch_new_experiment", {"model_candidate":"model_candidate"}),
    "resume_training": ("resume_training", "continue_from_checkpoint", {"training_duration":"training_duration"}),
    "rollback": ("rollback", "rollback_to_checkpoint", {"checkpoint_resume_point":"checkpoint_resume_point"}),
    "replace_or_mix_dataset": ("replace_or_mix_dataset", "branch_new_experiment", {"dataset_mixture":"dataset_mixture"}),
    "change_tokenizer": ("change_tokenizer", "branch_new_experiment", {"tokenizer":"tokenizer"}),
    "stop_experiment": ("stop", "abort_run", {"experiment_budget":"termination_decision"}),
}

DIAGNOSIS_DOMAINS = {
    "tokenizer":("tokenizer",), "evaluation_integrity":("evaluation_integrity",),
    "data_coverage":("data_coverage",), "data_format_or_wrapper":("data_format_or_wrapper",),
    "runtime_or_numerics":("runtime_or_system", "numerical_instability"),
    "undertraining":("undertraining",), "overfitting":("overfitting",),
    "checkpoint_integrity":("checkpoint_integrity",),
    "model_family_limitation":("model_family_limitation",),
    "inconclusive":("inconclusive",), "observability_limit":("inconclusive",),
}


def planner_selection(answer, receipt_ref: str) -> ModelInterventionSelection:
    try:
        kind, action, variables = PLANNER_DECISIONS[answer.decision]
        primary = variables[answer.primary_variable]
    except KeyError as exc:
        raise ValueError("ambiguous or unsupported model Planner decision/variable") from exc
    if answer.action != action:
        raise ValueError("model Planner decision/action mismatch")
    return ModelInterventionSelection(kind, primary, answer.confidence, list(answer.evidence_refs),
                                      receipt_ref, answer.rationale, answer.to_dict())
