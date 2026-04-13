from __future__ import annotations

from dataclasses import dataclass

from hephaestus.schemas.runtime_event import RuntimeEvent, RuntimeEventCategory


@dataclass(slots=True)
class DryRunBackend:
    name: str = "dry_run"

    def acquire_dataset(self, run_id: str) -> dict[str, object]:
        return {
            "dataset_id": f"dataset-{run_id}",
            "source_identity": "approved:synthetic:v1",
            "license": "internal",
            "quality_score": 0.92,
            "risks": ["synthetic_bias"],
            "total_examples": 1200,
            "source_ids": ["synthetic://seed_corpus"],
            "artifact_ref": f"artifacts/{run_id}/dataset_manifest.json",
        }

    def preprocess(self, run_id: str) -> dict[str, object]:
        return {
            "operations": ["normalize", "filter", "deduplicate", "chunk"],
            "processed_dataset_ref": f"artifacts/{run_id}/processed_dataset.jsonl",
            "dropped_examples": 24,
        }

    def launch_training(self, run_id: str) -> dict[str, object]:
        return {
            "checkpoint_candidates": [
                {"checkpoint_ref": f"artifacts/{run_id}/ckpt-100", "probe_score": 0.61},
                {"checkpoint_ref": f"artifacts/{run_id}/ckpt-200", "probe_score": 0.68},
            ],
            "intermediate_eval": {
                "probe_score": 0.68,
                "toxicity": 0.04,
                "probe_ref": f"artifacts/{run_id}/probe_step_200.json",
                "metrics_ref": f"artifacts/{run_id}/metrics_step_200.json",
                "deterministic_ref": f"artifacts/{run_id}/det_checks_step_200.json",
            },
        }

    def runtime_events(self, run_id: str) -> list[RuntimeEvent]:
        return [
            RuntimeEvent(event_id=f"{run_id}-evt-1", run_id=run_id, step=100, category=RuntimeEventCategory.STATUS, message="running", payload_ref=None),
            RuntimeEvent(event_id=f"{run_id}-evt-2", run_id=run_id, step=200, category=RuntimeEventCategory.METRIC, message="probe=0.68", payload_ref=f"artifacts/{run_id}/metrics_step_200.json"),
            RuntimeEvent(event_id=f"{run_id}-evt-3", run_id=run_id, step=200, category=RuntimeEventCategory.PROBE, message="generated probe outputs", payload_ref=f"artifacts/{run_id}/probe_step_200.json"),
            RuntimeEvent(event_id=f"{run_id}-evt-4", run_id=run_id, step=200, category=RuntimeEventCategory.DETERMINISTIC_CHECK, message="deterministic checks pass", payload_ref=f"artifacts/{run_id}/det_checks_step_200.json"),
        ]
