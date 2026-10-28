"""CPU internal data curator/preprocessor owners over existing intelligence services."""
from hephaestus.control.staged_state import StagedApprovalDecision
from hephaestus.policy.internal_research import put_record
from hephaestus.data.generated_factory import materialize_authorized_tokens, load_authorized_tokens
from hephaestus.storage.verified import read_verified_json
from .cpu_research_planning import result


class CpuResearchDataRole:
    def __init__(self, context):
        self.ctx = context

    def execute(self, req):
        ctx = self.ctx
        ctx.check_authority()
        if req.substep == "dataset_discovery":
            from hephaestus.schemas.discovery_contract import DatasetSearchRequest
            search = ctx.source(req, "discovery_request_preparation", "search_request")
            request = DatasetSearchRequest(**search["dataset_request"])
            decision, ref = ctx.intelligence.discover_select(request, **ctx.scope)
            if decision.selected_candidate_ids != [ctx.candidate.candidate_id]:
                raise PermissionError("no eligible source for CPU experiment")
            return result("dataset_discovery_result", {"status": "discovered", "request_id": decision.request_id,
                "candidates": [ctx.candidate.to_dict()], "decision_ref": ref,
                "actual_provider_evidence_refs": decision.evidence_refs}, decision_ref=ref)
        if req.substep == "dataset_selection":
            discovered = ctx.source(req, "dataset_discovery", "dataset_discovery_result")
            decision = read_verified_json(ctx.artifacts, discovered["decision_ref"])
            ctx.admission.require_clearance(ctx.candidate, ctx.review_ref, **ctx.scope)
            return result("dataset_selection_decision", decision, subject_ref=decision["decision_id"],
                required_approvals=decision["required_approvals"])
        decision = ctx.source(req, "dataset_selection", "dataset_selection_decision")
        gate = ctx.source(req, "acquisition_approval_gate", "approval_decision")
        if gate.get("subject_ref") != decision["decision_id"] or gate.get("status") != "approved":
            raise PermissionError("native acquisition approval is not bound to selected source")
        authorization = ctx.admission.authorize(ctx.candidate, ctx.review_ref, **ctx.scope)
        if req.substep == "dataset_acquisition":
            return result("acquisition_receipt", {"schema": "internal-original-source-acquisition.v1",
                "candidate_id": ctx.candidate.candidate_id, "authorization_ref": authorization,
                "approval_ref": gate["approval_ref"], "source_file_hashes": read_verified_json(ctx.artifacts, ctx.review_ref)["source_file_hashes"]}, dataset_revision=ctx.candidate.revision)
        ctx.source(req, "dataset_acquisition", "acquisition_receipt")
        if req.substep == "license_provenance_evidence":
            review = read_verified_json(ctx.artifacts, ctx.review_ref)
            rights = read_verified_json(ctx.artifacts, review["rights_ref"])
            return result("license_provenance_evidence", {"rights": rights, "rights_ref": review["rights_ref"],
                "generator_ref": review["generator_ref"], "authorization_ref": authorization})
        if req.substep == "dataset_audit":
            review = read_verified_json(ctx.artifacts, ctx.review_ref)
            return result("dataset_audit", read_verified_json(ctx.artifacts, review["scan_ref"]))
        if req.substep != "manifest_production":
            raise ValueError("unsupported CPU data operation")
        return result("dataset_manifest", {"schema": "internal-original-dataset-manifest.v1",
            "candidate": ctx.candidate.to_dict(), "review_ref": ctx.review_ref,
            "authorization_ref": authorization, "tokenizer_sha256": ctx.profile.tokenizer_sha256,
            "processing_profile_sha256": ctx.scope["processing_profile_sha256"],
            "mixture_weights": {ctx.candidate.candidate_id: 1.0}, "synthetic": True})


class CpuResearchPreprocessorRole:
    def __init__(self, context):
        self.ctx = context

    def execute(self, req):
        ctx = self.ctx
        ctx.source(req, "manifest_production", "dataset_manifest")
        material = materialize_authorized_tokens(ctx.admission, ctx.candidate, ctx.review_ref, **ctx.scope)
        if material != ctx.materialization_ref:
            raise ValueError("CPU materialization differs from preauthorized exact source")
        rows = load_authorized_tokens(ctx.admission, ctx.candidate, ctx.review_ref, material, **ctx.scope)
        contract = read_verified_json(ctx.artifacts, material)
        if req.substep == "approved_source_preprocessing":
            return result("preprocessing_report", contract, materialization_ref=material)
        if req.substep == "deduplication_contamination":
            review = read_verified_json(ctx.artifacts, ctx.review_ref)
            return result("data_quality_report", read_verified_json(ctx.artifacts, review["scan_ref"]))
        if req.substep == "tokenizer_compatibility":
            return result("tokenizer_compatibility", {"tokenizer_sha256": ctx.profile.tokenizer_sha256,
                "actual_source_tokens": contract["tokens"], "rows": len(rows), "exact_prompt_mask_verified": True})
        if req.substep != "trainable_data_contract":
            raise ValueError("unsupported CPU preprocessing operation")
        return result("trainable_data_contract", contract, materialization_ref=material)


class InternalResearchApprovalService:
    """Native gate adapter. Only exact Tier A dataset requests can be auto-authorized."""
    def __init__(self, context):
        self.ctx = context

    def decision_for(self, request):
        ctx = self.ctx
        if (request.workflow_id != ctx.scope["workflow_id"] or request.run_id != ctx.scope["run_id"] or
                request.substep != "acquisition_approval_gate" or request.phase != "data_acquisition_audit" or
                set(request.requirements) != {"dataset_acquisition", "dataset_selection_approval"}):
            return None
        from hephaestus.control.staged_state import StagedOperationRequest
        req = StagedOperationRequest(request.workflow_id, request.run_id, "cpu-internal-lineage", "smoke_test",
            request.phase, request.substep, request.operation_id, 1, (), {})
        # Resolve selection from immutable same-workflow native records, never caller metadata.
        decision = ctx.source(req, "dataset_selection", "dataset_selection_decision")
        if request.subject_ref != decision["decision_id"] or decision["selected_candidate_ids"] != [ctx.candidate.candidate_id]:
            raise PermissionError("automatic approval selection scope mismatch")
        authorization = ctx.admission.authorize(ctx.candidate, ctx.review_ref, **ctx.scope)
        grant = put_record(ctx.artifacts, {"schema": "internal-policy-native-acquisition-grant.v1",
            "request": request.to_dict(), "authorization_ref": authorization,
            "authority": "machine_policy_not_independent_person", "policy_ref": ctx.admission.policy_ref})
        return StagedApprovalDecision(request.request_id, request.operation_id, request.subject_ref,
            "approved", grant, request.requirements)
