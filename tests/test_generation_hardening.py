from __future__ import annotations

import json
from pathlib import Path

from hephaestus.generation import DeterministicFakeGenerationBackend, EvaluationGenerationService
from hephaestus.schemas.experiment_contract import TrainingRunHandle


def test_empty_generation_is_partial_failure(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    handoff = checkpoint / "loading_instructions.json"
    handoff.write_text(
        json.dumps(
            {
                "backend": "transformers_causal_lm",
                "model_artifact_ref": str(checkpoint / "model"),
                "tokenizer_artifact_ref": str(checkpoint / "tokenizer"),
                "trust_remote_code": False,
            }
        ),
        encoding="utf-8",
    )
    run = TrainingRunHandle(
        run_id="empty-output-run",
        experiment_id="experiment-empty-output",
        backend_id="transformers_causal_lm",
        status="completed",
        checkpoint_refs=[str(checkpoint)],
        metadata={"generation_handoff_ref": str(handoff)},
    )
    backend = DeterministicFakeGenerationBackend(fallback=lambda _run, _task: "   ")
    result = EvaluationGenerationService(tmp_path / "artifacts", backend).generate(run)

    assert not result.report.completed
    assert result.report.completion_status == "partial"
    assert [issue.code for issue in result.report.issues] == ["generation_backend_failed", "generation_samples_missing"]
    assert "semantic_evaluation" in result.run_handle.metadata
    assert result.run_handle.metadata["semantic_evaluation"]["completion_status"] == "partial"
