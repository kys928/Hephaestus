"""GPU-scoped native context with a fresh independently authorized execution lease.

Preparation is CPU-only and may precede human approval. The original CPU
preparation timestamp must not be treated as a GPU spending lease, but the
new lease may only be constructed after the actual operator-signed grant.
"""
from dataclasses import dataclass, fields
from datetime import datetime,timezone

from hephaestus.production.cpu_research_context import CpuResearchContext
from hephaestus.production.signed_cuda_grants import SignedOriginalCudaGrantService


@dataclass
class AuthorizedCudaResearchContext(CpuResearchContext):
    approved_gpu_execution_start: datetime = None
    approved_gpu_max_seconds: int = 0

    @classmethod
    def from_signed_approved(cls, source, *, grants, scope):
        if (type(source) is not CpuResearchContext
                or type(grants) is not SignedOriginalCudaGrantService
                or grants.authorized.get("workflow_id")!=scope.workflow_id
                or grants.authorized.get("run_id")!=scope.run_id
                or grants.authorized.get("operator_id")!=scope.trusted_approval_authority
                or grants.authorized.get("source_policy_ref")!=source.admission.policy_ref
                or type(scope.maximum_gpu_seconds) is not int
                or not 1<=scope.maximum_gpu_seconds<=grants.authorized["maximum_gpu_seconds"]):
            raise PermissionError("only independently signed GPU execution may reset deadline")
        return cls(**{field.name:getattr(source,field.name) for field in fields(CpuResearchContext)},
                   approved_gpu_execution_start=datetime.now(timezone.utc),
                   approved_gpu_max_seconds=scope.maximum_gpu_seconds)

    def check_deadline(self):
        start=self.approved_gpu_execution_start
        if not isinstance(start,datetime) or start.tzinfo is None:
            raise PermissionError("CUDA deadline requires a valid signed execution lease")
        elapsed=(datetime.now(timezone.utc)-start).total_seconds()
        if elapsed<0 or elapsed>self.approved_gpu_max_seconds:
            raise PermissionError("independently approved GPU time budget expired")
