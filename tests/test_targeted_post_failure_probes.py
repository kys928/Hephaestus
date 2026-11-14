from __future__ import annotations

from hephaestus.diagnosis.probes import (
    analyze_dataset_task_coverage,
    analyze_training_dynamics,
)


def _metrics(*, improving: bool = True, malformed_lr: bool = False):
    rows = []
    for step in range(5, 101, 5):
        loss = 9.0 - (0.018 * step if improving else 0.002 * min(step, 20))
        if step <= 10:
            lr = 0.0005 * (step / 10)
        else:
            lr = 0.0005 * max(0.0, (100 - step) / 90)
        if malformed_lr and step == 50:
            lr = 0.001
        rows.append(
            {
                "step": step,
                "training_loss": loss,
                "learning_rate": lr,
                "gradient_norm": 0.8,
                "epoch": step * 0.0001457779,
            }
        )
    return rows


def _config():
    return {
        "max_steps": 100,
        "warmup_steps": 10,
        "learning_rate": 0.0005,
        "scheduler": "linear",
    }


def _eval_pack():
    return {
        "generation_probes": [{"prompt": "Reply with exactly: alpha beta gamma."}],
        "continuation_prompts": [{"prompt": "The observatory lost power."}],
        "structure_tests": [{"prompt": "Return JSON with answer and confidence."}],
        "repetition_checks": [{"prompt": "Explain in one sentence why evidence helps."}],
        "length_termination_checks": [{"prompt": "State one benefit in at most twelve words."}],
    }


def test_learning_curve_and_budget_support_undertraining_without_scheduler_blame() -> None:
    evidence, measurements = analyze_training_dynamics(
        _metrics(), _config(), source_ref="metrics.jsonl"
    )

    assert evidence["undertraining_detected"] is True
    assert evidence["training_budget_exhausted"] is True
    assert evidence["optimizer_stable"] is True
    assert evidence["scheduler_misconfigured"] is False
    assert measurements["tail_still_improving"] is True
    assert measurements["final_epoch_fraction"] < 0.05


def test_plateau_does_not_invent_undertraining_signal() -> None:
    evidence, _ = analyze_training_dynamics(
        _metrics(improving=False), _config(), source_ref="metrics.jsonl"
    )

    assert evidence["undertraining_detected"] is False
    assert evidence["training_budget_exhausted"] is False


def test_scheduler_trace_mismatch_is_explicit_optimizer_scheduler_evidence() -> None:
    evidence, measurements = analyze_training_dynamics(
        _metrics(malformed_lr=True), _config(), source_ref="metrics.jsonl"
    )

    assert evidence["scheduler_misconfigured"] is True
    assert evidence["optimizer_stable"] is False
    assert measurements["scheduler_conforms_to_recorded_shape"] is False


def test_raw_text_only_training_data_exposes_task_coverage_gap() -> None:
    rows = ({"text": f"encyclopedic continuation {index}"} for index in range(100))
    evidence, measurements = analyze_dataset_task_coverage(
        rows, _eval_pack(), source_ref="trainable.jsonl"
    )

    assert evidence["data_coverage_gap"] is True
    assert measurements["prompt_target_rows"] == 0
    assert measurements["structured_target_rows"] == 0


def test_supervised_instruction_and_structured_targets_cover_frozen_task_forms() -> None:
    rows = [
        {"prompt": "Reply briefly", "target": "alpha beta gamma", "text": "alpha beta gamma"},
        {"prompt": "Return JSON", "target": '{"answer": "Mars", "confidence": 1.0}'},
    ]
    evidence, measurements = analyze_dataset_task_coverage(
        rows, _eval_pack(), source_ref="trainable.jsonl"
    )

    assert evidence["data_coverage_gap"] is False
    assert evidence["data_coverage_verified"] is True
    assert measurements["prompt_target_rows"] == 2
    assert measurements["structured_target_rows"] == 1



def test_embedded_prompt_target_in_text_is_counted_without_inventing_structure() -> None:
    rows = [
        {"text":"<|prompt|>\\nAnswer briefly\\n<|target|>\\nMars"},
        {"text":"<|prompt|>\\nReturn JSON\\n<|target|>\\nnot JSON"},
        {"text":"<|prompt|>\\nMissing response\\n<|target|>\\n"},
        {"text":"<|prompt|>\\nUnfinished only"},
        {"text":"<|target|>\\nWrong order\\n<|prompt|>\\nWrong"},
        {"text":"<|prompt|>\\nRepeated <|prompt|>\\n<|target|>\\nIncomplete"},
    ]
    evidence, measurements = analyze_dataset_task_coverage(
        iter(rows), _eval_pack(), source_ref="synthetic-representation-contract-test",
    )
    assert measurements["coverage_probe_version"] == 2
    assert measurements["rows_scanned"] == 6
    assert measurements["explicit_prompt_target_rows"] == 0
    assert measurements["embedded_marker_pair_rows"] == 5
    assert measurements["embedded_valid_prompt_target_rows"] == 2
    assert measurements["embedded_empty_or_malformed_rows"] == 3
    assert measurements["prompt_target_rows"] == 2
    assert measurements["structured_target_rows"] == 0
    assert evidence["data_coverage_gap"] is True
    assert measurements["supervised_instruction_coverage_present"] is True
    assert measurements["supervised_structure_coverage_present"] is False


def test_embedded_json_object_targets_count_but_unparseable_json_does_not() -> None:
    rows = [
        {"text":'<|prompt|>\\nRespond in JSON\\n<|target|>\\n{"answer":"Mars","confidence":1.0}'},
        {"text":'<|prompt|>\\nRespond in JSON\\n<|target|>\\n{"broken":}'},
        {"text":'<|prompt|>\\nRespond in JSON\\n<|target|>\\n[1,2]'},
    ]
    evidence, measurements = analyze_dataset_task_coverage(
        rows, _eval_pack(), source_ref="synthetic-json-contract-test",
    )
    assert measurements["embedded_valid_prompt_target_rows"] == 3
    assert measurements["embedded_structured_target_rows"] == 1
    assert measurements["structured_target_rows"] == 1
    assert evidence["data_coverage_gap"] is False


def test_duplicate_field_and_embedded_supervision_does_not_double_count_a_row() -> None:
    evidence, measurements = analyze_dataset_task_coverage(
        [{"prompt":"Answer", "target":'{"ok":true}',
          "text":'<|prompt|>\\nAnswer\\n<|target|>\\n{"ok":true}'}],
        _eval_pack(), source_ref="synthetic-overlap-contract-test",
    )
    assert measurements["explicit_prompt_target_rows"] == 1
    assert measurements["embedded_valid_prompt_target_rows"] == 1
    assert measurements["prompt_target_rows"] == 1
    assert measurements["structured_target_rows"] == 1
    assert evidence["data_coverage_gap"] is False


def test_marker_presence_is_not_sufficient_if_prompt_or_target_is_empty() -> None:
    evidence, measurements = analyze_dataset_task_coverage(
        [
            {"text":"<|prompt|>\\n\\n<|target|>\\nanswer"},
            {"text":"<|prompt|>\\nprompt\\n<|target|>\\n"},
            {"prompt":"", "target":"nonempty", "text":"unrelated"},
        ],
        _eval_pack(), source_ref="synthetic-empty-contract-test",
    )
    assert measurements["embedded_marker_pair_rows"] == 2
    assert measurements["embedded_valid_prompt_target_rows"] == 0
    assert measurements["explicit_prompt_target_rows"] == 1
    assert measurements["prompt_target_rows"] == 0
    assert measurements["structured_target_rows"] == 0
    assert evidence["data_coverage_gap"] is True
