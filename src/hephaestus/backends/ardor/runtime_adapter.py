from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from hephaestus.backends.base import BackendRunResult, PreparedBackendJob
from hephaestus.backends.ardor.launcher import ArdorLaunchOutcome
from hephaestus.backends.ardor.runtime_contract import normalize_ardor_runtime_contract
from hephaestus.runtime.event_stream import events_from_process_output
from hephaestus.schemas.runtime_event import RuntimeEvent, RuntimeEventCategory


@dataclass(slots=True)
class ArdorRuntimeAdapter:
    def normalize_run(self, prepared_job: PreparedBackendJob, outcome: ArdorLaunchOutcome) -> BackendRunResult:
        if outcome.status != "launched":
            return BackendRunResult(
                run_id=prepared_job.run_id,
                status="failed",
                events=[RuntimeEvent(event_id=f"{prepared_job.run_id}-launch-failure", run_id=prepared_job.run_id, step=0, category=RuntimeEventCategory.INCIDENT, message=f"ardor_launch_failure status={outcome.status} detail={outcome.stderr}", payload_ref=outcome.contract_ref)],
                artifact_refs=[ref for ref in [outcome.contract_ref] if ref],
                checkpoint_candidates=[],
                intermediate_eval={},
            )
        events = events_from_process_output(prepared_job.run_id, outcome.stdout, outcome.stderr)
        contract_ref = str(outcome.contract_ref or "")
        contract_path = Path(contract_ref) if contract_ref else None
        if contract_path is None or not contract_path.exists():
            events.append(RuntimeEvent(event_id=f"{prepared_job.run_id}-missing-contract", run_id=prepared_job.run_id, step=0, category=RuntimeEventCategory.INCIDENT, message="ardor_missing_runtime_contract", payload_ref=contract_ref or None))
            return BackendRunResult(prepared_job.run_id, "failed", events, [], [], {})
        try:
            payload = json.loads(contract_path.read_text())
        except json.JSONDecodeError:
            events.append(RuntimeEvent(event_id=f"{prepared_job.run_id}-malformed-contract", run_id=prepared_job.run_id, step=0, category=RuntimeEventCategory.INCIDENT, message="ardor_malformed_output_contract", payload_ref=contract_ref))
            return BackendRunResult(prepared_job.run_id, "failed", events, [contract_ref], [], {})

        normalized = normalize_ardor_runtime_contract(payload, contract_ref=contract_ref)
        if normalized.get("run_id") and normalized.get("run_id") != prepared_job.run_id:
            events.append(RuntimeEvent(event_id=f"{prepared_job.run_id}-run-id-mismatch", run_id=prepared_job.run_id, step=0, category=RuntimeEventCategory.INCIDENT, message=f"ardor_contract_run_id_mismatch:{normalized.get('run_id')}", payload_ref=contract_ref))
            return BackendRunResult(prepared_job.run_id, "failed", events, [contract_ref], [], {})
        if normalized["contract_integrity_level"] == "legacy":
            events.append(RuntimeEvent(event_id=f"{prepared_job.run_id}-legacy-contract", run_id=prepared_job.run_id, step=0, category=RuntimeEventCategory.STATUS, message="ardor_legacy_contract_detected", payload_ref=contract_ref))
        for warning in normalized.get("warnings", []):
            category = RuntimeEventCategory.STATUS
            if str(warning).startswith("malformed_checkpoint_candidate"):
                category = RuntimeEventCategory.INCIDENT
            events.append(RuntimeEvent(event_id=f"{prepared_job.run_id}-warning-{hashlib.sha256(str(warning).encode()).hexdigest()[:12]}", run_id=prepared_job.run_id, step=0, category=category, message=f"ardor_contract_warning:{warning}", payload_ref=contract_ref))
        if normalized["contract_integrity_level"] == "insufficient":
            events.append(RuntimeEvent(event_id=f"{prepared_job.run_id}-insufficient-contract", run_id=prepared_job.run_id, step=0, category=RuntimeEventCategory.INCIDENT, message="ardor_insufficient_contract", payload_ref=contract_ref))
            return BackendRunResult(prepared_job.run_id, "failed", events, [contract_ref], [], {})

        status = self._map_status(prepared_job.run_id, str(normalized.get("status") or ""), events, contract_ref)
        artifacts = dict(normalized.get("artifacts") or {})
        checkpoint_candidates = list(normalized.get("checkpoint_candidates") or [])
        intermediate_eval = {k: str(artifacts.get(k, "") or "") for k in ("metrics_ref", "probe_ref", "deterministic_ref", "runtime_log_ref")}
        self._normalize_metrics_artifact(prepared_job.run_id, intermediate_eval, normalized, events)

        artifact_refs = [contract_ref]
        for key in ("metrics_ref", "probe_ref", "deterministic_ref", "runtime_log_ref", "dataset_manifest_ref", "training_recipe_ref", "tokenizer_ref", "architecture_config_ref", "eval_report_ref", "eval_pack_ref"):
            ref = str(artifacts.get(key, "") or "")
            if ref:
                artifact_refs.append(ref)
        artifact_refs.extend(str(item.get("checkpoint_ref", "")) for item in checkpoint_candidates if str(item.get("checkpoint_ref", "")))

        status = self._validate_artifacts(prepared_job.run_id, status, artifacts, artifact_refs, checkpoint_candidates, events)
        if outcome.returncode not in (0, None):
            events.append(RuntimeEvent(event_id=f"{prepared_job.run_id}-nonzero-exit", run_id=prepared_job.run_id, step=0, category=RuntimeEventCategory.INCIDENT, message=f"ardor_returncode:{outcome.returncode}", payload_ref=contract_ref))
            status = "failed"
        for name in ("probe_score", "toxicity"):
            if name in intermediate_eval:
                events.append(RuntimeEvent(event_id=f"{prepared_job.run_id}-metric-{name}", run_id=prepared_job.run_id, step=0, category=RuntimeEventCategory.METRIC, message=f"{name}={intermediate_eval[name]}", payload_ref=intermediate_eval.get("metrics_ref") or contract_ref))
        return BackendRunResult(prepared_job.run_id, status, events, artifact_refs, checkpoint_candidates, intermediate_eval)

    def _map_status(self, run_id: str, ardor_status: str, events: list[RuntimeEvent], payload_ref: str) -> str:
        mapped = {"succeeded": "completed", "failed": "failed", "partial": "failed", "unsupported": "failed"}
        if ardor_status in mapped:
            if ardor_status in {"partial", "unsupported"}:
                events.append(RuntimeEvent(event_id=f"{run_id}-{ardor_status}", run_id=run_id, step=0, category=RuntimeEventCategory.INCIDENT, message=f"ardor_runtime_state_{ardor_status}", payload_ref=payload_ref))
            return mapped[ardor_status]
        events.append(RuntimeEvent(event_id=f"{run_id}-unsupported-state", run_id=run_id, step=0, category=RuntimeEventCategory.INCIDENT, message=f"ardor_unsupported_runtime_state:{ardor_status or 'missing'}", payload_ref=payload_ref))
        return "failed"

    def _normalize_metrics_artifact(self, run_id: str, intermediate_eval: dict[str, object], normalized: dict[str, object], events: list[RuntimeEvent]) -> None:
        metrics_ref = str(intermediate_eval.get("metrics_ref", "") or "")
        metrics = dict(normalized.get("metrics") or {})
        if metrics_ref and Path(metrics_ref).exists():
            try:
                payload = json.loads(Path(metrics_ref).read_text())
                if isinstance(payload, dict):
                    metrics.update(dict(payload.get("metrics", {})) if isinstance(payload.get("metrics"), dict) else payload)
            except json.JSONDecodeError:
                events.append(RuntimeEvent(event_id=f"{run_id}-malformed-metrics", run_id=run_id, step=0, category=RuntimeEventCategory.INCIDENT, message="ardor_malformed_metrics_artifact", payload_ref=metrics_ref))
        if "probe_score" in metrics:
            intermediate_eval["probe_score"] = float(metrics.get("probe_score", 0.0))
        if "toxicity" not in metrics:
            metrics["toxicity"] = 0.0
        intermediate_eval["toxicity"] = float(metrics.get("toxicity", 0.0))
        if metrics_ref and Path(metrics_ref).exists() and "probe_score" in metrics:
            Path(metrics_ref).write_text(json.dumps({"probe_score": float(metrics["probe_score"]), "toxicity": float(metrics["toxicity"])}, indent=2))

    def _validate_artifacts(self, run_id: str, status: str, artifacts: dict[str, object], artifact_refs: list[str], checkpoint_candidates: list[dict[str, object]], events: list[RuntimeEvent]) -> str:
        for key in ("metrics_ref", "deterministic_ref"):
            if status == "completed" and not str(artifacts.get(key, "") or ""):
                events.append(RuntimeEvent(event_id=f"{run_id}-missing-{key}", run_id=run_id, step=0, category=RuntimeEventCategory.INCIDENT, message=f"ardor_missing_{key}", payload_ref=None))
                status = "failed"
        optional = ("tokenizer_ref", "architecture_config_ref", "eval_report_ref", "eval_pack_ref")
        for key in optional:
            if not str(artifacts.get(key, "") or ""):
                events.append(RuntimeEvent(event_id=f"{run_id}-missing-opt-{key}", run_id=run_id, step=0, category=RuntimeEventCategory.STATUS, message=f"ardor_missing_optional_{key}", payload_ref=None))
        if status == "completed" and not checkpoint_candidates:
            events.append(RuntimeEvent(event_id=f"{run_id}-missing-checkpoints", run_id=run_id, step=0, category=RuntimeEventCategory.INCIDENT, message="ardor_missing_checkpoint_refs", payload_ref=None))
            status = "failed"
        for ref in artifact_refs:
            if ref and not Path(ref).exists():
                events.append(RuntimeEvent(event_id=f"{run_id}-missing-artifact-{Path(ref).name}", run_id=run_id, step=0, category=RuntimeEventCategory.INCIDENT, message=f"ardor_missing_artifact_ref:{ref}", payload_ref=ref))
                status = "failed"
        return status
