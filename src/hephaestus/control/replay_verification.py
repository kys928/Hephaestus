"""Read-only replay/reproducibility verification for persisted runs."""

from __future__ import annotations

from pathlib import Path

from hephaestus.schemas.replay_verification import ReplayVerificationReport
from hephaestus.state.artifact_index import ArtifactIndex
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.manifest_store import ManifestStore
from hephaestus.state.memory_store import MemoryStore
from hephaestus.state.report_store import ReportStore
from hephaestus.state.run_store import RunStore


def _as_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _as_ref(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _stable_unique(values: list[object]) -> list[str]:
    seen: set[str] = set()
    refs: list[str] = []
    for value in values:
        ref = _as_ref(value)
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def _find_eval_report(
    reports: list[dict[str, object]], run_id: str, eval_report_id: str | None
) -> dict[str, object] | None:
    if eval_report_id:
        for report in reversed(reports):
            if (
                str(report.get("kind") or "") == "eval_report"
                and str(report.get("eval_id") or "") == eval_report_id
            ):
                return report
    for report in reversed(reports):
        if (
            str(report.get("kind") or "") == "eval_report"
            and str(report.get("run_id") or "") == run_id
        ):
            return report
    return None


def _checkpoint_refs_match(
    checkpoint_ref: str | None,
    replay_metadata: dict[str, object],
    decision_metadata: dict[str, object],
    eval_report: dict[str, object] | None,
) -> tuple[bool, list[str]]:
    if not checkpoint_ref:
        return True, []
    available = [
        ("decision.metadata.checkpoint_ref", decision_metadata.get("checkpoint_ref")),
        ("replay_metadata.checkpoint_ref", replay_metadata.get("checkpoint_ref")),
        ("replay_metadata.selected_checkpoint_ref", replay_metadata.get("selected_checkpoint_ref")),
    ]
    if eval_report:
        resolution = _as_dict(eval_report.get("checkpoint_resolution"))
        available.append(
            (
                "eval_report.checkpoint_resolution.selected_checkpoint_ref",
                resolution.get("selected_checkpoint_ref"),
            )
        )
    mismatches: list[str] = []
    observed_any = False
    for label, value in available:
        ref = _as_ref(value)
        if not ref:
            continue
        observed_any = True
        if ref != checkpoint_ref:
            mismatches.append(f"checkpoint_ref_mismatch:{label}")
    if not observed_any:
        # Older replay metadata may only record that the run checkpoint reference must match.
        return True, []
    return not mismatches, mismatches


def verify_run_replay(state_root: str | Path, run_id: str) -> ReplayVerificationReport:
    """Verify whether a stored run's decision context can be reconstructed from evidence.

    The verifier is intentionally read-only: it only loads existing stores and returns a
    deterministic report derived from persisted records. It does not inspect model
    outputs beyond persisted references and does not infer missing evidence.
    """

    root = Path(state_root)
    run_store = RunStore(root)
    lineage_store = LineageStore(root)
    decision_store = DecisionStore(root)
    manifest_store = ManifestStore(root)
    report_store = ReportStore(root)
    artifact_index = ArtifactIndex(root)
    memory_store = MemoryStore(root)

    run = run_store.get(run_id)
    if not run:
        return ReplayVerificationReport(
            run_id=run_id,
            lineage_id=None,
            status="missing",
            checked_at="",
            missing_evidence=["run_record"],
            summary="Run record is missing; replay verification cannot begin.",
        )

    lineage_id = _as_ref(run.get("lineage_id"))
    replay_metadata = _as_dict(run.get("replay_metadata"))
    checkpoint_ref = _as_ref(run.get("checkpoint_ref"))
    manifest_id = _as_ref(run.get("data_manifest_id"))
    eval_report_id = _as_ref(run.get("eval_report_id"))
    decision_id = f"dec-{run_id}-exit"

    missing: list[str] = []
    warnings: list[str] = []
    partial_reasons: list[str] = []

    if not replay_metadata:
        missing.append("replay_metadata")

    lineage = lineage_store.get_current(lineage_id) if lineage_id else None
    if not lineage_id:
        missing.append("lineage_id")
    elif lineage is None:
        partial_reasons.append("lineage_state")
        warnings.append("lineage_state_missing")

    manifest = manifest_store.get(manifest_id) if manifest_id else None
    if manifest_id and manifest is None:
        missing.append("data_manifest")
    elif not manifest_id:
        partial_reasons.append("data_manifest_id")
        warnings.append("run_has_no_data_manifest_id")

    reports = report_store.all()
    eval_report = _find_eval_report(reports, run_id=run_id, eval_report_id=eval_report_id)
    if eval_report_id and eval_report is None:
        missing.append("eval_report")
    elif not eval_report_id:
        partial_reasons.append("eval_report_id")
        warnings.append("run_has_no_eval_report_id")

    decision = decision_store.get(decision_id)
    decision_metadata = _as_dict((decision or {}).get("metadata"))
    if decision is None:
        missing.append("judge_exit_decision")
    else:
        if not isinstance(decision_metadata.get("promotion_gate_report"), dict):
            missing.append("judge_exit_decision.metadata.promotion_gate_report")
        if not isinstance(decision_metadata.get("action_boundary"), dict):
            missing.append("judge_exit_decision.metadata.action_boundary")

    checkpoint_match, checkpoint_mismatches = _checkpoint_refs_match(
        checkpoint_ref=checkpoint_ref,
        replay_metadata=replay_metadata,
        decision_metadata=decision_metadata,
        eval_report=eval_report,
    )
    if not checkpoint_match:
        missing.extend(checkpoint_mismatches)

    checkpoint_content_hash = _as_ref(replay_metadata.get("checkpoint_content_hash"))
    content_hash_available = _as_bool(replay_metadata.get("content_hash_available")) or bool(checkpoint_content_hash)
    requires_content_hash_match = _as_bool(replay_metadata.get("requires_content_hash_match"))
    if requires_content_hash_match and not checkpoint_content_hash:
        missing.append("checkpoint_content_hash")
    if checkpoint_ref and not content_hash_available:
        partial_reasons.append("checkpoint_content_hash")
        warnings.append("checkpoint_content_hash_unavailable")

    artifacts = [row for row in artifact_index.all() if str(row.get("run_id") or "") == run_id]
    memories = memory_store.list_for_run(run_id)

    evidence_values: list[object] = [
        run.get("run_id"),
        lineage_id,
        manifest_id,
        eval_report_id,
        decision_id if decision is not None else None,
        checkpoint_ref,
    ]
    if manifest:
        evidence_values.append(manifest.get("artifact_ref"))
    if eval_report:
        if isinstance(eval_report.get("intermediate_artifact_refs"), list):
            evidence_values.extend(eval_report.get("intermediate_artifact_refs", []))
        scorecard = _as_dict(eval_report.get("deterministic_scorecard"))
        if isinstance(scorecard.get("evidence_refs"), list):
            evidence_values.extend(scorecard.get("evidence_refs", []))
    if decision and isinstance(decision.get("evidence_refs"), list):
        evidence_values.extend(decision.get("evidence_refs", []))
    evidence_values.extend(row.get("ref") for row in artifacts)
    for memory in memories:
        refs = memory.get("evidence_refs", [])
        if isinstance(refs, list):
            evidence_values.extend(refs)

    missing_sorted = sorted(set(missing))
    warnings_sorted = sorted(set(warnings))
    replay_scope = _as_ref(replay_metadata.get("replay_scope")) or "unknown"
    confidence_ceiling_raw = decision_metadata.get("confidence_ceiling")
    confidence_ceiling = float(confidence_ceiling_raw) if isinstance(confidence_ceiling_raw, (int, float)) else None

    if missing_sorted:
        status = "insufficient"
        summary = "Replay verification found missing critical decision evidence."
    elif partial_reasons:
        status = "partial"
        summary = (
            "Decision context is partially reproducible from persisted references, "
            "but byte-identical replay is not proven."
        )
    else:
        status = "reproducible"
        summary = (
            "Decision context is reproducible from persisted evidence and "
            "recorded checkpoint integrity metadata."
        )

    return ReplayVerificationReport(
        run_id=run_id,
        lineage_id=lineage_id,
        status=status,
        checked_at=_as_ref(run.get("completed_at")) or _as_ref(run.get("started_at")) or "",
        evidence_refs=_stable_unique(evidence_values),
        missing_evidence=missing_sorted,
        warnings=warnings_sorted,
        replay_scope=replay_scope,
        checkpoint_ref=checkpoint_ref,
        checkpoint_content_hash=checkpoint_content_hash,
        content_hash_available=content_hash_available,
        requires_content_hash_match=requires_content_hash_match,
        manifest_id=manifest_id,
        eval_report_id=eval_report_id,
        decision_id=decision_id if decision is not None else None,
        confidence_ceiling=confidence_ceiling,
        summary=summary,
    )
