"""Read legacy memory and derive experiment history from verified durable records."""
from dataclasses import dataclass
import hashlib
from hephaestus.schemas.memory_record import MemoryRecord
from hephaestus.storage.base import ArtifactStore, StateRepository
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from .query import Query


@dataclass(slots=True)
class ExperimentMemoryQuery(Query):
    repository: StateRepository
    artifacts: ArtifactStore

    def _experiment_memories(self, lineage_id: str | None):
        rows = []
        for collection in ("production_integration_records", "staged_workflow_records"):
            for row in self.repository.all(collection):
                if not row.get("artifact_ref"):
                    continue  # Unsealed historical records are not scientific failure proof.
                source = read_verified_json(self.artifacts, str(row["artifact_ref"]))
                if source != {k:v for k,v in row.items() if k != "artifact_ref"}:
                    raise ValueError("experiment memory source record differs from immutable evidence")
                rows.append((str(row["kind"]), dict(row["payload"]), str(row["artifact_ref"])))
        experiments = {str(p["experiment_id"]):p for kind,p,_ in rows
                       if kind == "experiment_proposal" and p.get("experiment_id")}
        interventions = {str(p["intervention_id"]):p for kind,p,_ in rows
                         if kind == "intervention_proposal" and p.get("intervention_id")}
        memories = {}
        for kind, comparison, ref in rows:
            if kind != "experiment_comparison":
                continue
            experiment = experiments.get(str(comparison.get("experiment_id")))
            if experiment is None or (lineage_id and experiment.get("lineage_id") != lineage_id):
                continue
            intervention = interventions.get(str(experiment.get("intervention_id")), {})
            intervention_kind = intervention.get("intervention_kind") or experiment.get("metadata", {}).get("intervention_kind")
            if not intervention_kind:
                continue
            outcome = str(comparison.get("primary_outcome"))
            failed = outcome in {"regressed", "scientific_rejection"} or comparison.get("deterministic_gate_status") == "failed"
            memory_id = "mem-" + hashlib.sha256(canonical_bytes([ref, experiment["experiment_id"]])).hexdigest()[:24]
            memories[memory_id] = MemoryRecord(memory_id,
                "known_dead_end" if failed else "experiment_outcome", "experiment_comparison",
                str(comparison["comparison_id"]), lineage_id=str(experiment["lineage_id"]),
                run_id=str(experiment["run_id"]), stage_name=str(experiment["stage_name"]),
                summary=f"Controlled {intervention_kind} experiment outcome: {outcome}.",
                evidence_refs=list(dict.fromkeys([*experiment.get("required_evidence", []),
                    *intervention.get("required_inputs", []), ref])),
                tags=[str(intervention_kind), str(experiment["primary_variable"])],
                confidence=1.0, metadata={"intervention_kind":intervention_kind,
                    "primary_variable":experiment["primary_variable"], "outcome":outcome,
                    "baseline_ref":experiment.get("baseline_ref"),
                    "controlled_variables":experiment.get("controlled_variables", {}),
                    "causal_failure_claim":False}).to_dict()
        return list(memories.values())

    def dead_ends_for_lineage(self, lineage_id: str, limit: int = 20):
        rows = [*super(ExperimentMemoryQuery, self).dead_ends_for_lineage(lineage_id, limit),
                *(r for r in self._experiment_memories(lineage_id) if r["memory_type"] == "known_dead_end")]
        return rows[-limit:]

    def similar_failure_patterns(self, lineage_id=None, tags=None, limit=20):
        rows = [*super(ExperimentMemoryQuery, self).similar_failure_patterns(lineage_id, tags, limit),
                *(r for r in self._experiment_memories(lineage_id) if r["memory_type"] == "known_dead_end")]
        if tags:
            rows = [r for r in rows if set(tags).intersection(r["tags"])]
        return rows[-limit:]

    def intervention_history(self, lineage_id: str, limit: int = 20):
        return [*super(ExperimentMemoryQuery, self).intervention_history(lineage_id, limit),
                *self._experiment_memories(lineage_id)][-limit:]
