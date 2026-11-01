"""Cold verification and real fixed-experiment replay of the public CPU proof."""
import json
from pathlib import Path
import tempfile

from hephaestus.control.staged_state import StagedWorkflowState,WORKFLOW_STATE_COLLECTION
from hephaestus.development.cpu_autonomy_benchmark import load
from hephaestus.control.cpu_autonomy_context import CpuAutonomyContext
from hephaestus.production.staged_trace import verify_completed_staged_trace
from hephaestus.schemas.prospective_model_binding import ProspectiveModelBinding
from hephaestus.policy.cpu_autonomy import learning_rate,accept_candidate
from hephaestus.evaluation.prospective_pipeline import digest,file_digest,verify_model_binding
from hephaestus.evaluation.cpu_autonomy import measure
from hephaestus.storage.verified import canonical_bytes,read_verified_json
from hephaestus.state.cpu_autonomy_store import CpuAutonomyStore
from hephaestus.state.lineage_store import LineageStore


def replay_training(job_path,checkpoint):
    import torch
    from safetensors.torch import load_file
    from hephaestus.training.hf_worker import run
    job=json.loads(Path(job_path).read_text())
    with tempfile.TemporaryDirectory(prefix="hephaestus-cold-training-") as folder:
        path=Path(folder); job["artifact_root"]=str(path)
        (path/"job.json").write_bytes(canonical_bytes(job))
        if run(path/"job.json")!=0:
            raise ValueError("cold actual training replay failed")
        replay=path/f"checkpoint_step_{job['training_config']['max_steps']}"
        original=load_file(str(Path(checkpoint)/"model"/"adapter_model.safetensors"))
        repeated=load_file(str(replay/"model"/"adapter_model.safetensors"))
        if set(original)!=set(repeated) or any(not torch.equal(original[k],repeated[k]) for k in original):
            raise ValueError("cold optimizer replay did not reproduce every adapter tensor exactly")
        a=torch.load(Path(checkpoint)/"training_state.pt",map_location="cpu",weights_only=True)
        b=torch.load(replay/"training_state.pt",map_location="cpu",weights_only=True)
        for key in ("step","examples_processed","tokens_processed","batches_processed","last_training_loss","last_gradient_norm","config_fingerprint"):
            if a[key]!=b[key]:
                raise ValueError("cold training dose/metric replay differs: "+key)
        def equal(left,right):
            if isinstance(left,torch.Tensor):
                return isinstance(right,torch.Tensor) and torch.equal(left,right)
            if isinstance(left,dict):
                return isinstance(right,dict) and set(left)==set(right) and all(equal(left[k],right[k]) for k in left)
            if isinstance(left,(tuple,list)):
                return isinstance(right,type(left)) and len(left)==len(right) and all(equal(x,y) for x,y in zip(left,right,strict=True))
            return left==right
        for key in ("optimizer_state","scheduler_state","python_rng_state","torch_rng_state"):
            if not equal(a[key],b[key]):
                raise ValueError("cold optimizer/scheduler/RNG state replay differs: "+key)


def verify_program(root,program_id,*,reexecute_models=True,reexecute_training=True):
    import torch
    torch.set_num_threads(1)
    rt,m,anchor,plan=load(root,read_only=True)
    for path,expected in {**m["input_files"],**m["method_files"]}.items():
        if file_digest(path)!=expected:
            raise ValueError("cold benchmark input or method source changed")
    final=CpuAutonomyStore(rt.state_repository,rt.artifact_store).final(program_id)
    if final is None or final["status"]!="completed":
        raise ValueError("cold benchmark lacks a completed final publication")
    result=read_verified_json(rt.artifact_store,final["result_ref"])
    state=rt.loop_state.get(program_id)
    if (state.status!="stopped" or state.cycle_index!=3 or state.stop_reason!="maximum_cycles_reached"
            or result["actual_optimizer_updates"]!=12 or result["native_completed_operations"]!=135
            or result["external_certification"] is not False or result["production_promotion"] is not False):
        raise ValueError("cold program outcome/count/scope differs from actual execution")
    context=CpuAutonomyContext(rt,state,plan,m,anchor,result["policy"],result["seed"],3)
    parent=context.binding(); preceding=[]
    memories=rt.planner.memory_query.intervention_history(state.lineage_id)
    for index,cycle in enumerate(result["cycles"],start=1):
        event=cycle["event"]
        if event["cycle_id"]!=f"{program_id}:cycle:{index}":
            raise ValueError("cold cycle order differs")
        payload=rt.state_repository.get_latest(WORKFLOW_STATE_COLLECTION,"workflow_id",event["evidence"]["workflow_id"])
        workflow=StagedWorkflowState.from_dict(payload)
        trace=verify_completed_staged_trace(workflow,rt.state_repository,rt.artifact_store)
        if trace!=event["evidence"]["operation_trace_ref"]:
            raise ValueError("cold 45-operation trace differs from completion event")
        prior=[mem for mem in memories if mem["run_id"] in preceding]
        rate,_=learning_rate(plan,result["policy"],index,prior)
        proposal=cycle["proposal"]
        if proposal["training_constraints"]["learning_rate"]!=rate or proposal["baseline_ref"]!=digest(parent.identity()):
            raise ValueError("cold policy decision or parent checkpoint differs from preceding evidence")
        candidate=context.binding(event["checkpoint_ref"],event["run_id"])
        verify_model_binding(candidate,context.design)
        metrics=read_verified_json(rt.artifact_store,cycle["metrics_ref"])
        if metrics["steps"]!=4 or metrics["examples_processed"]!=4:
            raise ValueError("cold matched training budget differs")
        from hephaestus.control.staged_state import StagedOperationRequest
        req=StagedOperationRequest(workflow.workflow_id,workflow.run_id,workflow.lineage_id,workflow.stage_name,
                                  "evaluator","semantic_comparison","cold-read-only",1,(),{})
        comparison=context.source(req,"semantic_comparison","experiment_comparison")
        if comparison!=cycle["comparison"]:
            raise ValueError("cold comparison differs from original immutable workflow")
        measurements=[]
        for step,binding in (("baseline_generation",parent),("candidate_generation",candidate)):
            record=context.source(req,step,"generation_report")
            measured=read_verified_json(rt.artifact_store,record["measurement_ref"])
            measurements.append(measured)
            if measured["generation"]["model_identity"]!=binding.identity():
                raise ValueError("cold generation names a different checkpoint")
            if reexecute_models and measure(binding,m["development"],context.design)!=measured:
                raise ValueError("cold development inference/NLL reexecution differs")
        accepted=accept_candidate(*measurements,plan.development_nll_margin)
        if accepted!=comparison["metadata"]["accepted_by_frozen_development_rule"]:
            raise ValueError("cold Judge acceptance differs from actual development quality")
        if reexecute_training:
            replay_training(candidate.job_path,candidate.checkpoint_path)
        if comparison["metadata"]["accepted_by_frozen_development_rule"]:
            parent=candidate
        preceding.append(event["run_id"])
    if parent.identity()!=ProspectiveModelBinding.from_dict(result["selected_binding"]).identity():
        raise ValueError("cold final selection differs from all preceding Judge transitions")
    if LineageStore(rt.settings.state_root).get_current(state.lineage_id)!=result["lineage"]:
        raise ValueError("cold lineage truth differs from final publication")
    if final["identity"]!=digest({"manifest":anchor,"program":program_id,"selected":parent.identity(),"test":digest(m["test"])}):
        raise ValueError("cold final test exposure identity differs")
    measured=read_verified_json(rt.artifact_store,result["test_measurement_ref"])
    if reexecute_models and measure(parent,m["test"],context.design)!=measured:
        raise ValueError("cold final fixed-model inference/NLL reexecution differs")
    if measured["mean_target_nll"]!=result["test_mean_target_nll"] or measured["exact_correct"]!=result["test_exact_correct"]:
        raise ValueError("cold reported test quality differs from original outputs")
    from hephaestus.state.staged_retry_store import COLLECTION
    retries=[r for r in rt.state_repository.all(COLLECTION) if r["cycle_key"].startswith(program_id+":") and r.get("failure_ref")]
    if len(retries)!=1 or result["repair_count"]!=1:
        raise ValueError("cold expected one bounded injected-failure recovery")
    failure=read_verified_json(rt.artifact_store,retries[0]["failure_ref"])
    proof=read_verified_json(rt.artifact_store,failure["optimizer_progress_ref"])
    handle=read_verified_json(rt.artifact_store,failure["handle_ref"])["payload"]
    failed=rt.state_repository.get_latest(WORKFLOW_STATE_COLLECTION,"workflow_id",failure["failed_workflow_id"])
    if (proof["optimizer_steps"]!=0 or proof["optimizer_update_started"] is not False
            or handle["status"]!="failed" or failed["status"]!="terminal_failure" or failed["completion_marker"]):
        raise ValueError("cold failed predecessor was lost or falsely completed")
    decision=read_verified_json(rt.artifact_store,retries[0]["decision_ref"])
    if decision["classification"]["category"]!="process_crash" or decision["recommendation"]["action_kind"]!="restart_bounded_job":
        raise ValueError("cold injected failure diagnosis or safe intervention differs")
    return {"program_id":program_id,"verified_operations":135,"actual_optimizer_updates":12,
        "failed_predecessor_preserved":True,"all_fixed_models_reexecuted":reexecute_models,
        "all_optimizer_tensors_reproduced":reexecute_training,"external_independent_certification":False}
