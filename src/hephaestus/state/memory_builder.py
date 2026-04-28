from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from hephaestus.schemas.memory_record import MemoryRecord


def _memory_id(
    memory_type: str,
    source_kind: str,
    source_id: str,
    lineage_id: str | None,
    run_id: str | None,
    stage_name: str | None,
    summary: str,
    tags: list[str],
) -> str:
    payload = {
        "memory_type": memory_type,
        "source_kind": source_kind,
        "source_id": source_id,
        "lineage_id": lineage_id,
        "run_id": run_id,
        "stage_name": stage_name,
        "summary": summary,
        "tags": sorted(str(item) for item in tags),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"mem-{digest}"


def _severity_from_gate_report(gate_report: dict[str, Any], fallback: str = "major") -> str:
    for gate in gate_report.get("gates", []):
        if isinstance(gate, dict) and bool(gate.get("blocking")) and str(gate.get("severity")) == "critical":
            return "critical"
    return fallback


def _related_intervention_ids(lineage_state: dict[str, object] | None) -> list[str]:
    state = lineage_state or {}
    related: list[str] = []
    for item in state.get("major_interventions", []):
        if not isinstance(item, dict):
            continue
        run_id = str(item.get("run_id") or "")
        target_ref = str(item.get("target_checkpoint_ref") or "")
        if run_id:
            related.append(run_id)
        if target_ref:
            related.append(target_ref)
    return related


def build_memory_records_from_run(
    run_id: str,
    run_record: dict[str, object] | None,
    lineage_state: dict[str, object] | None,
    decisions: list[dict[str, object]],
    reports: list[dict[str, object]],
    manifests: list[dict[str, object]],
) -> list[MemoryRecord]:
    run = run_record or {}
    lineage = lineage_state or {}
    lineage_id = str(lineage.get("lineage_id") or run.get("lineage_id") or "") or None
    stage_name = str(run.get("stage_name") or lineage.get("stage_name") or "") or None
    created_at = str(run.get("completed_at") or run.get("started_at") or "") or None

    records: list[MemoryRecord] = []

    for decision in decisions:
        metadata = dict(decision.get("metadata", {}))
        gate_report = dict(metadata.get("promotion_gate_report", {}))
        blocking_failures = [str(item) for item in gate_report.get("blocking_failures", metadata.get("blocking_failures", []))]
        if blocking_failures:
            severity = _severity_from_gate_report(gate_report, fallback="major")
            for failure in blocking_failures:
                tags = [
                    "promotion_block",
                    f"action:{metadata.get('requested_action', decision.get('action', 'unknown'))}",
                ]
                memory_id = _memory_id(
                    memory_type="promotion_block",
                    source_kind="decision",
                    source_id=str(decision.get("decision_id") or f"dec-{run_id}-exit"),
                    lineage_id=lineage_id,
                    run_id=run_id,
                    stage_name=stage_name,
                    summary=failure,
                    tags=tags,
                )
                records.append(
                    MemoryRecord(
                        memory_id=memory_id,
                        memory_type="promotion_block",
                        source_kind="decision",
                        source_id=str(decision.get("decision_id") or f"dec-{run_id}-exit"),
                        lineage_id=lineage_id,
                        run_id=run_id,
                        stage_name=stage_name,
                        created_at=created_at,
                        severity=severity,
                        summary=f"Promotion blocked: {failure}",
                        tags=tags,
                        related_ids=[str(decision.get("decision_id") or "")],
                        confidence=0.95,
                        metadata={
                            "requested_action": metadata.get("requested_action", decision.get("action")),
                            "gate_report_id": gate_report.get("run_id"),
                            "blocking_failure": failure,
                        },
                    )
                )

        if bool(metadata.get("promotion_allowed")) and str(metadata.get("effective_action", "")) == "promote_checkpoint":
            memory_id = _memory_id(
                memory_type="successful_intervention",
                source_kind="decision",
                source_id=str(decision.get("decision_id") or ""),
                lineage_id=lineage_id,
                run_id=run_id,
                stage_name=stage_name,
                summary="Promotion allowed after gate evaluation.",
                tags=["promotion", "successful_intervention"],
            )
            records.append(
                MemoryRecord(
                    memory_id=memory_id,
                    memory_type="successful_intervention",
                    source_kind="decision",
                    source_id=str(decision.get("decision_id") or ""),
                    lineage_id=lineage_id,
                    run_id=run_id,
                    stage_name=stage_name,
                    created_at=created_at,
                    severity="info",
                    summary="Promotion-like action passed gating and remained effective.",
                    tags=["promotion", "successful_intervention"],
                    related_ids=[str(metadata.get("effective_action"))],
                    confidence=0.8,
                    metadata={"promotion_allowed": True},
                )
            )

    failure_counter = Counter(str(item) for item in lineage.get("recent_failures", []))
    repeated_failures = [name for name, count in failure_counter.items() if name and count >= 2]
    for failure in repeated_failures:
        memory_id = _memory_id(
            memory_type="repeated_failure",
            source_kind="lineage_state",
            source_id=str(lineage_id or "lineage"),
            lineage_id=lineage_id,
            run_id=run_id,
            stage_name=stage_name,
            summary=failure,
            tags=["repeated_failure"],
        )
        records.append(
            MemoryRecord(
                memory_id=memory_id,
                memory_type="repeated_failure",
                source_kind="lineage_state",
                source_id=str(lineage_id or "lineage"),
                lineage_id=lineage_id,
                run_id=run_id,
                stage_name=stage_name,
                created_at=created_at,
                severity="major",
                summary=f"Repeated lineage failure observed: {failure}",
                tags=["repeated_failure"],
                related_ids=[failure],
                confidence=0.9,
                metadata={"count": int(failure_counter[failure])},
            )
        )

    status = str(lineage.get("status", ""))
    if status in {"poisoned", "deprecated", "archived"}:
        memory_id = _memory_id(
            memory_type="known_dead_end",
            source_kind="lineage_state",
            source_id=str(lineage_id or "lineage"),
            lineage_id=lineage_id,
            run_id=run_id,
            stage_name=stage_name,
            summary=f"Lineage status is {status}.",
            tags=["known_dead_end", f"status:{status}"],
        )
        records.append(
            MemoryRecord(
                memory_id=memory_id,
                memory_type="known_dead_end",
                source_kind="lineage_state",
                source_id=str(lineage_id or "lineage"),
                lineage_id=lineage_id,
                run_id=run_id,
                stage_name=stage_name,
                created_at=created_at,
                severity="critical" if status == "poisoned" else "major",
                summary=f"Lineage is marked {status}; treat as known dead end.",
                tags=["known_dead_end", f"status:{status}"],
                related_ids=_related_intervention_ids(lineage_state),
                confidence=0.95,
                metadata={"lineage_status": status},
            )
        )

    for manifest in manifests:
        integrity = str(manifest.get("manifest_integrity_level") or "insufficient")
        warnings = [str(item) for item in manifest.get("warnings", [])]
        if integrity in {"insufficient", "reference_only"} or warnings:
            manifest_id = str(manifest.get("manifest_id") or run.get("data_manifest_id") or "unknown-manifest")
            memory_id = _memory_id(
                memory_type="data_issue",
                source_kind="manifest",
                source_id=manifest_id,
                lineage_id=lineage_id,
                run_id=run_id,
                stage_name=stage_name,
                summary=f"Manifest integrity {integrity}",
                tags=["data_manifest", f"integrity:{integrity}"],
            )
            records.append(
                MemoryRecord(
                    memory_id=memory_id,
                    memory_type="data_issue",
                    source_kind="manifest",
                    source_id=manifest_id,
                    lineage_id=lineage_id,
                    run_id=run_id,
                    stage_name=stage_name,
                    created_at=created_at,
                    severity="major" if integrity == "insufficient" else "warning",
                    summary=f"Data manifest integrity is '{integrity}'.",
                    tags=["data_manifest", f"integrity:{integrity}"],
                    evidence_refs=[str(manifest.get("artifact_ref") or "")],
                    related_ids=[manifest_id],
                    confidence=0.9 if integrity == "insufficient" else 0.75,
                    metadata={"warnings": warnings},
                )
            )

    for report in reports:
        if str(report.get("kind")) != "eval_report":
            continue
        eval_pack_level = str(report.get("eval_pack_integrity_level") or "insufficient")
        scorecard_level = str(report.get("scorecard_integrity_level") or "insufficient")
        deterministic_scorecard = dict(report.get("deterministic_scorecard", {}))
        insufficient = eval_pack_level in {"insufficient", "reference_only", "inline_unhashed"} or scorecard_level in {
            "insufficient",
            "reference_only",
            "inline_unhashed",
        }
        missing_det = not deterministic_scorecard
        if insufficient or missing_det:
            eval_id = str(report.get("eval_id") or f"eval-{run_id}")
            memory_id = _memory_id(
                memory_type="eval_issue",
                source_kind="report",
                source_id=eval_id,
                lineage_id=lineage_id,
                run_id=run_id,
                stage_name=stage_name,
                summary=f"Eval integrity pack={eval_pack_level} scorecard={scorecard_level}",
                tags=["eval_issue", f"eval_pack:{eval_pack_level}", f"scorecard:{scorecard_level}"],
            )
            records.append(
                MemoryRecord(
                    memory_id=memory_id,
                    memory_type="eval_issue",
                    source_kind="report",
                    source_id=eval_id,
                    lineage_id=lineage_id,
                    run_id=run_id,
                    stage_name=stage_name,
                    created_at=created_at,
                    severity="major" if missing_det or "insufficient" in {eval_pack_level, scorecard_level} else "warning",
                    summary="Eval evidence has integrity limitations or missing deterministic scorecard.",
                    tags=["eval_issue", f"eval_pack:{eval_pack_level}", f"scorecard:{scorecard_level}"],
                    related_ids=[str(report.get("eval_pack_id") or "")],
                    confidence=0.85,
                    metadata={
                        "eval_pack_integrity_level": eval_pack_level,
                        "scorecard_integrity_level": scorecard_level,
                        "deterministic_scorecard_present": bool(deterministic_scorecard),
                    },
                )
            )

    for intervention in lineage.get("major_interventions", []):
        if not isinstance(intervention, dict):
            continue
        intervention_type = str(intervention.get("type") or "")
        if intervention_type not in {"rollback", "branch"}:
            continue
        memory_type = "rollback_event" if intervention_type == "rollback" else "branch_event"
        summary = f"Lineage intervention recorded: {intervention_type}."
        tags = [memory_type, intervention_type]
        source_id = str(intervention.get("run_id") or lineage_id or "lineage")
        memory_id = _memory_id(
            memory_type=memory_type,
            source_kind="lineage_state",
            source_id=source_id,
            lineage_id=lineage_id,
            run_id=run_id,
            stage_name=stage_name,
            summary=summary,
            tags=tags,
        )
        records.append(
            MemoryRecord(
                memory_id=memory_id,
                memory_type=memory_type,
                source_kind="lineage_state",
                source_id=source_id,
                lineage_id=lineage_id,
                run_id=run_id,
                stage_name=stage_name,
                created_at=created_at,
                severity="info",
                summary=summary,
                tags=tags,
                related_ids=[str(intervention.get("target_checkpoint_ref") or "")],
                confidence=0.9,
                metadata={"intervention": intervention},
            )
        )

    deduped: dict[str, MemoryRecord] = {}
    for record in records:
        deduped[record.memory_id] = record
    return [deduped[key] for key in sorted(deduped)]
