"""Training-model selection and actual monitor; launch belongs to the real lifecycle adapter."""
import json
from pathlib import Path
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
from hephaestus.evaluation.prospective_pipeline import file_digest
from .cpu_autonomy_planning import result


class CpuAutonomyTrainingRole:
    def __init__(self,context): self.ctx=context

    def execute(self,req):
        c=self.ctx; c.check(req); m=c.manifest
        if req.substep=="model_discovery":
            config=json.loads((Path(m["model_path"])/"config.json").read_text())
            return result("model_discovery_result",{"model_path":m["model_path"],"model_revision":m["model_identity"],
                "config":config,"random_initialization_seed":c.plan.base_seed,"pretrained":False},model_revision=m["model_identity"])
        if req.substep=="model_selection":
            discovered=c.source(req,"model_discovery","model_discovery_result")
            return result("model_selection_decision",{"status":"selected","identity":discovered,"required_approvals":[]},
                subject_ref=c.manifest_ref,required_approvals=[],model_revision=m["model_identity"])
        raise ValueError("CPU model-selection owner cannot replace training lifecycle operations")


class CpuAutonomyMonitorRole:
    def __init__(self,context): self.ctx=context

    def execute(self,req):
        c=self.ctx; c.check(req)
        handle=c.runtime.training_lifecycle.status(req.run_id)
        if handle.status!="completed" or not handle.checkpoint_refs or not validate_checkpoint_manifest(handle.checkpoint_refs[-1])[0]:
            raise ValueError("CPU monitor requires actual completed verified training")
        metrics=json.loads(Path(handle.metrics_ref).read_text())
        if metrics["steps"]!=c.plan.steps_per_cycle or metrics["examples_processed"]!=4:
            raise ValueError("CPU training dose differs from frozen budget")
        facts={"run_id":req.run_id,"training_status":handle.status,"checkpoint_ref":handle.checkpoint_refs[-1],
            "metrics_ref":c.runtime.artifact_store.put_bytes(Path(handle.metrics_ref).read_bytes()).artifact_ref,
            "metrics_hash":file_digest(handle.metrics_ref),"optimizer_steps":metrics["steps"],"examples_processed":metrics["examples_processed"]}
        if req.substep=="runtime_observation":
            return result("runtime_observation",facts)
        if req.substep=="runtime_evidence":
            return result("runtime_evidence",facts,monitor_outcome="healthy")
        if req.substep=="bounded_recovery_advice":
            return result("recovery_decision",{"status":"not_required_completed_training","evidence":facts,"action_executed":False})
        if req.substep=="runtime_control_governance":
            return result("runtime_control_recommendation",{"recommendation":"retain_evidence_and_evaluate","training_completed":True,"action_executed":False})
        raise ValueError("unsupported CPU monitor operation")
