"""Read-only evidence adapters for the diagnosis service."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from hephaestus.schemas.diagnosis_contract import DiagnosisRequest

EvidenceRecord = Mapping[str, object]


@runtime_checkable
class EvidenceAdapter(Protocol):
    """Resolve evidence without changing source state."""

    def load(self, request: DiagnosisRequest) -> Sequence[EvidenceRecord]: ...


@dataclass(slots=True)
class MappingEvidenceAdapter:
    """Resolve request references from an in-memory fixture or application map."""

    records_by_ref: Mapping[str, object]

    def load(self, request: DiagnosisRequest) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        for ref in request.evidence_refs:
            value = self.records_by_ref.get(ref)
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                record = deepcopy(dict(candidate))
                record.setdefault("source_ref", ref)
                records.append(record)
        return records


@dataclass(slots=True)
class JsonReferenceEvidenceAdapter:
    """Load JSON/JSONL evidence references confined to a configured root."""

    root: Path

    def load(self, request: DiagnosisRequest) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        root = self.root.resolve()
        for ref in request.evidence_refs:
            path = (
                (root / ref).resolve()
                if not Path(ref).is_absolute()
                else Path(ref).resolve()
            )
            if path != root and root not in path.parents:
                continue
            if not path.is_file():
                continue
            for index, payload in enumerate(_read_records(path)):
                record = deepcopy(payload)
                suffix = f"#{index}" if path.suffix == ".jsonl" else ""
                record.setdefault("source_ref", f"{ref}{suffix}")
                records.append(record)
        return records


@dataclass(slots=True)
class StateEvidenceAdapter:
    """Read existing filesystem state without invoking write-capable stores."""

    root: Path

    def load(self, request: DiagnosisRequest) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        jsonl_sources = {
            "run_records.jsonl": "run_record",
            "runtime_events.jsonl": "runtime_event",
            "incidents.jsonl": "incident_record",
            "manifests.jsonl": "dataset_manifest",
            "reports.jsonl": "report",
            "decision_records.jsonl": "decision_record",
            "memory_records.jsonl": "memory_record",
            "artifact_index.jsonl": "artifact_index",
        }
        for filename, default_kind in jsonl_sources.items():
            path = self.root / filename
            if not path.is_file():
                continue
            for index, payload in enumerate(_read_records(path)):
                if not _belongs_to_request(payload, request):
                    continue
                record = deepcopy(payload)
                kind = _infer_kind(record, default_kind)
                record.setdefault("evidence_kind", kind)
                record.setdefault("source_ref", f"state/{filename}#{index}")
                records.append(record)

        lineage = _load_lineage(self.root, request.lineage_id)
        if lineage is not None:
            lineage.setdefault("evidence_kind", "lineage_state")
            lineage.setdefault("source_ref", f"state/lineage:{request.lineage_id}")
            records.append(lineage)
        return records


def _read_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            rows: list[dict[str, object]] = []
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
            return rows
        payload = json.load(handle)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _belongs_to_request(
    payload: Mapping[str, object], request: DiagnosisRequest
) -> bool:
    run_id = str(payload.get("run_id") or "")
    lineage_id = str(payload.get("lineage_id") or "")
    return (run_id == request.run_id) or (
        not run_id and lineage_id == request.lineage_id
    )


def _infer_kind(payload: Mapping[str, object], fallback: str) -> str:
    explicit = str(payload.get("evidence_kind") or payload.get("kind") or "")
    if explicit:
        return explicit
    if "eval_id" in payload or "deterministic_scorecard" in payload:
        return "eval_report"
    if "manifest_id" in payload and "datasets" in payload:
        return "dataset_manifest"
    return fallback


def _load_lineage(root: Path, lineage_id: str) -> dict[str, object] | None:
    all_path = root / "lineage_states.json"
    if all_path.is_file():
        rows = _read_records(all_path)
        if rows:
            candidate = rows[0].get(lineage_id)
            if isinstance(candidate, dict):
                return deepcopy(candidate)
    legacy_path = root / "lineage_state.json"
    if legacy_path.is_file():
        rows = _read_records(legacy_path)
        if rows and str(rows[0].get("lineage_id") or "") == lineage_id:
            return deepcopy(rows[0])
    return None
