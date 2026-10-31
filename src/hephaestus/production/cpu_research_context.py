"""Shared scientific IO context; phase ownership remains in separate role adapters."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from hephaestus.llm.cpu_causal_research import CpuCausalProposalScorer
from hephaestus.control.deterministic_role_boundary import EvidenceRecord, EvidenceRegistry, DeterministicRoleContext
from hephaestus.llm.role_contract import RoleTask, RoleEvidence
from hephaestus.policy.internal_research import put_record
from hephaestus.schemas.cpu_research import CpuResearchExecutionProfile
from hephaestus.storage.verified import read_verified_json
from hephaestus.training.cpu_research import verify_snapshot
from .staged_records import verified_workflow_payload


@dataclass
class CpuResearchContext:
    root: Path
    artifacts: object
    repository: object
    admission: object
    candidate: object
    review_ref: str
    materialization_ref: str
    scope: dict
    profile: CpuResearchExecutionProfile
    profile_ref: str
    snapshot: Path
    scorer: CpuCausalProposalScorer
    state: object
    data_report: dict
    intelligence: object
    search_request: object
    started_at: str
    campaign_config: object

    @property
    def profile_protocol(self):
        return self.admission.policy().protocol_ref

    def check_deadline(self):
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(self.started_at)).total_seconds()
        if elapsed > self.profile.maximum_cpu_seconds:
            raise PermissionError("bounded CPU research wall-time budget expired; renew explicitly")

    def check_authority(self, *, for_execution=True):
        if for_execution:
            self.check_deadline()
        self.profile.validate()
        policy = self.admission.policy()
        if (self.profile.policy_ref != self.admission.policy_ref or self.profile.tokenizer_sha256 != policy.tokenizer_sha256 or
                self.profile.maximum_training_steps > policy.maximum_training_steps or
                self.profile.maximum_cpu_seconds > policy.maximum_cpu_seconds):
            raise PermissionError("CPU profile exceeds internal policy authority")
        if type(self.scorer) is not CpuCausalProposalScorer or self.scorer.snapshot_dir != self.snapshot:
            raise PermissionError("real pinned CPU causal inference implementation required")
        if next(self.scorer.model.parameters()).device.type != "cpu":
            raise PermissionError("CPU profile cannot silently switch execution device")
        verify_snapshot(self.snapshot, self.profile.snapshot_hashes)
        return policy

    def source(self, req, substep, kind):
        return verified_workflow_payload(self.repository, self.artifacts, req, substep=substep, kind=kind)

    def role_proposal(self, role, req, payload):
        ref = put_record(self.artifacts, {"schema": "cpu-role-observed-domain-source.v1",
            "run_id": req.run_id, "lineage_id": req.lineage_id, "workflow_id": req.workflow_id,
            "phase": req.phase, "substep": req.substep, "domain_observation": payload})
        cached = self.state.get("role-" + req.operation_id, ref)
        if cached:
            return cached
        self.check_authority()
        registry = EvidenceRegistry({ref: EvidenceRecord(ref, "observed_internal_domain_result", req.run_id, req.lineage_id,
            sha256=ref[7:], observed_sha256=ref[7:])})
        context = DeterministicRoleContext(req.run_id, req.lineage_id, req.stage_name)
        task = RoleTask(role, "Review actual bounded internal prospective research evidence; approvals remain policy-owned.",
            (RoleEvidence(ref, str(payload)[:200]),), req.run_id, req.lineage_id, req.stage_name)
        result = self.scorer.propose(task, registry, context)
        self.state.put("role-" + req.operation_id, ref, result)
        return result
