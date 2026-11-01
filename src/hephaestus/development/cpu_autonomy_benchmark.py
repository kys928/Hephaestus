"""Public tiny CPU experiment: real native cycles, matched policies, no paid resource."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from hephaestus.control.cpu_autonomy import compose_cpu_autonomy
from hephaestus.control.cpu_autonomy_context import CpuAutonomyContext
from hephaestus.data.generated_factory import generate_original_splits
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.schemas.cpu_autonomy import CpuAutonomyPlan
from hephaestus.production.composition import ProductionCompositionRoot,ProductionCompositionSettings
from hephaestus.production.loop import ProductionLoopRunner
from hephaestus.production.staged_driver import StagedProductionCycleDriver
from hephaestus.state.cpu_autonomy_store import CpuAutonomyStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.execution_ownership import execution_owner
from hephaestus.storage.verified import canonical_bytes,read_verified_json
from hephaestus.evaluation.prospective_pipeline import digest,file_digest
from hephaestus.evaluation.cpu_autonomy import measure,compare_policies
from .prospective_rehearsal import create_tiny_model,prepare_cpu_proposal


def runtime(root):
    return ProductionCompositionRoot(ProductionCompositionSettings(
        state_root=root/"state",artifact_root=root/"artifacts",maximum_training_steps=4,
        enable_dataset_network=False,enable_model_network=False)).build()


def prepare(root):
    import torch
    torch.set_num_threads(1)
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=False)
    plan=CpuAutonomyPlan()
    tokenizer,count=create_tiny_model(root/"base")
    splits,_=generate_original_splits(GeneratedDataSpec(plan.data_seed))
    families=("arithmetic","json","symbolic","reasoning")
    subset=lambda name,n:[nexts[i] for i in range(n) for f in families
                          if (nexts:=[r for r in splits[name] if r["family"]==f])]
    train,development,test=subset("train",1),subset("development",2),subset("evaluation",4)
    source=root/"public-source.json"; source.write_bytes(canonical_bytes({"train":train,"development":development,"test":test}))
    proposal=prepare_cpu_proposal(root/"prepared",run_id="bounded-common",seed=1117,
        model_path=root/"base",tokenizer=tokenizer,parameter_count=count,rows=train)
    constraints=proposal.training_constraints
    recipe={"schema":"cpu-autonomy-finite-recipe.v1","steps_per_cycle":4,"rank":2,"alpha":4,
        "dropout":.1,"target_modules":["c_attn"],"learning_rate_grid":list(plan.learning_rate_grid),
        "optimizer_state":"reset_on_cross_run_adapter_warm_start","initialization":"random_gpt2_37472_parameters"}
    rt=runtime(root)
    recipe_ref=rt.artifact_store.put_bytes(canonical_bytes(recipe)).artifact_ref
    constraints.update(data_contract_ref=constraints["trainable_data_contract_ref"],training_recipe_ref=recipe_ref)
    paths=[source,Path(constraints["trainable_data_contract_ref"]),Path(constraints["processed_dataset_ref"]),Path(constraints["processing_evidence_ref"])]
    from hephaestus.backends.hf_causal_lm import directory_content_identity
    from importlib import import_module
    modules=("schemas.cpu_autonomy","policy.cpu_autonomy","control.cpu_autonomy","control.cpu_autonomy_context",
        "roles.cpu_autonomy_planning","roles.cpu_autonomy_data","roles.cpu_autonomy_execution","roles.cpu_autonomy_evaluation",
        "evaluation.cpu_autonomy","evaluation.cpu_autonomy_verification","development.cpu_autonomy_benchmark","training.hf_worker","training.hf_lifecycle",
        "production.staged_driver","production.staged_retry","production.actions","evaluation.prospective_pipeline",
        "storage.sqlite","state.evidence_store","production.composition","control.staged_autonomous")
    method_files={str(Path(import_module("hephaestus."+name).__file__).resolve()):file_digest(import_module("hephaestus."+name).__file__)
                  for name in modules}
    manifest={"schema":"public-cpu-autonomy-manifest.v1","plan":json.loads(canonical_bytes(plan.to_dict())),"method_files":method_files,
        "source_path":str(source),"train":train,"development":development,"test":test,
        "input_files":{str(p):file_digest(p) for p in paths},"model_path":str(root/"base"),
        "model_identity":directory_content_identity(root/"base"),"training_constraints":constraints,
        "fault_plan":"first cycle first attempt: real CPU bootstrap failure after model/tokenizer preparation, before optimizer",
        "outcomes":["final test target NLL","final exact JSON accuracy","successful repairs","correct injected-failure diagnosis",
            "repeated failed attempts","candidate regression rate","actual optimizer updates","wall time","steps to development NLL gain .1"],
        "scope":"public already-exposed original examples; engineering comparison, not independent holdout science",
        "specialist_model_contributions":"not exercised; finite deterministic Hephaestus policy only"}
    ref=CpuAutonomyStore(rt.state_repository,rt.artifact_store).freeze("public-tiny-cpu-autonomy",manifest)
    (root/"manifest.json").write_bytes(canonical_bytes({"manifest_ref":ref}))
    return manifest,ref


def load(root,*,read_only=False):
    root=Path(root).resolve()
    if read_only:
        from hephaestus.control.cpu_autonomy_context import readback_runtime
        rt=readback_runtime(root)
    else:
        rt=runtime(root)
    anchor=json.loads((root/"manifest.json").read_text())["manifest_ref"]
    manifest=read_verified_json(rt.artifact_store,anchor)
    retained=rt.state_repository.get_latest("cpu_autonomy_manifests","benchmark_id","public-tiny-cpu-autonomy")
    if retained is None or retained["manifest_ref"]!=anchor:
        raise PermissionError("CPU benchmark manifest anchor differs")
    return rt,manifest,anchor,CpuAutonomyPlan.from_dict(manifest["plan"])


def run_program(root,policy,seed):
    root=Path(root).resolve(); rt,m,anchor,plan=load(root)
    if policy not in plan.policies or seed not in plan.seeds:
        raise ValueError("program is outside the predeclared matched comparison")
    program=f"cpu-{policy}-{seed}"; lineage="lineage-"+program
    with execution_owner(rt.state_repository,"cpu-benchmark-session",program):
        store=CpuAutonomyStore(rt.state_repository,rt.artifact_store)
        start=store.begin_program(program,digest({"manifest":anchor,"policy":policy,"seed":seed}))
        final=store.final(program)
        if final:
            if final["status"]!="completed":
                raise PermissionError("CPU final assessment has unknown outcome; do not reconsult")
            return read_verified_json(rt.artifact_store,final["result_ref"])
        if LineageStore(rt.settings.state_root).get_current(lineage) is None:
            LineageStore(rt.settings.state_root).set_current(LineageState(lineage,stage_name="bounded_cpu_research").to_dict())
        def factory(runtime,state,cycle):
            return compose_cpu_autonomy(CpuAutonomyContext(runtime,state,plan,m,anchor,policy,seed,cycle))
        restarted=False
        while True:
            state=ProductionLoopRunner(rt,StagedProductionCycleDriver(factory),maximum_cycles=plan.cycles).run(
                program_id=program,lineage_id=lineage,stage_name="bounded_cpu_research")
            if state.status!="waiting":
                break
            if (datetime.now(timezone.utc)-datetime.fromisoformat(start["started_at"])).total_seconds()>plan.maximum_program_seconds:
                from hephaestus.schemas.experiment_contract import TrainingControlRequest
                active=state.metadata.get("pending_cycle",{}).get("run_id")
                if active:
                    rt.training_lifecycle.control(TrainingControlRequest("budget-cancel-"+program,active,"cancel","cpu-benchmark","bounded wall-time exhausted"))
                raise TimeoutError("CPU benchmark program wall-time budget exhausted")
            if state.cycle_index>=1 and not restarted:
                rt=runtime(root); restarted=True
            time.sleep(.05)
        if state.status!="stopped" or state.cycle_index!=3 or state.stop_reason!="maximum_cycles_reached":
            raise RuntimeError("CPU native program did not complete its three cycles: "+json.dumps(state.to_dict()))
        context=CpuAutonomyContext(rt,state,plan,m,anchor,policy,seed,3)
        selected=context.parent()
        rt.state_repository.assert_integrity()
        identity=digest({"manifest":anchor,"program":program,"selected":selected.identity(),"test":digest(m["test"])})
        store=CpuAutonomyStore(rt.state_repository,rt.artifact_store)
        store.consume_final(program,identity)
        measured=measure(selected,m["test"],context.design)
        measurement_ref=rt.artifact_store.put_bytes(canonical_bytes(measured)).artifact_ref
        events=[r["payload"] for r in rt.state_repository.all("production_loop_events")
                if r.get("program_id")==program and r.get("kind")=="cycle_completed"]
        cycles=[]
        from hephaestus.control.staged_state import WORKFLOW_RECORD_COLLECTION
        for event in events:
            records=[r for r in rt.state_repository.all(WORKFLOW_RECORD_COLLECTION) if r.get("run_id")==event["run_id"]]
            comparison=next(r["payload"] for r in records if r["kind"]=="experiment_comparison")
            proposal=next(r["payload"] for r in records if r["kind"]=="experiment_proposal")
            handle=rt.training_lifecycle.status(event["run_id"])
            metrics=json.loads(Path(handle.metrics_ref).read_text())
            cycles.append({"event":event,"comparison":comparison,"proposal":proposal,
                "metrics_ref":rt.artifact_store.put_bytes(Path(handle.metrics_ref).read_bytes()).artifact_ref,
                "optimizer_updates":metrics["steps"],"examples_processed":metrics["examples_processed"]})
        from hephaestus.state.staged_retry_store import COLLECTION
        retries=[r for r in rt.state_repository.all(COLLECTION) if r["cycle_key"].startswith(program+":") and r.get("decision_ref")]
        diagnoses=[read_verified_json(rt.artifact_store,r["decision_ref"])["classification"]["category"] for r in retries]
        first_nll=cycles[0]["comparison"]["effect_summary"]["development_nll_before"]
        reaches=[(i+1)*plan.steps_per_cycle for i,c in enumerate(cycles)
                 if first_nll-c["comparison"]["effect_summary"]["development_nll_after"]>=plan.target_nll_improvement]
        regressions=sum(c["comparison"]["effect_summary"]["development_nll_after"]>
                        c["comparison"]["effect_summary"]["development_nll_before"] for c in cycles)
        result={"schema":"public-cpu-autonomy-program-result.v1","program_id":program,"manifest_ref":anchor,
            "policy":policy,"seed":seed,"cycles":cycles,"native_completed_operations":45*len(cycles),
            "actual_optimizer_updates":sum(c["optimizer_updates"] for c in cycles),"repair_count":state.recovery_attempts,
            "injected_failure_diagnoses":diagnoses,"restart_after_completed_cycle":restarted,
            "correct_injected_diagnosis_rate":sum(d=="process_crash" for d in diagnoses)/len(diagnoses) if diagnoses else 0.,
            "candidate_nll_regressions":regressions,"candidate_nll_regression_frequency":regressions/3,
            "nonimproving_candidates_proxy_not_causal_unnecessary_interventions":sum(
                not c["comparison"]["metadata"]["accepted_by_frozen_development_rule"] for c in cycles),
            "steps_to_development_nll_gain_point1":min(reaches) if reaches else None,
            "repeated_failed_experiments":max(0,len(retries)-1),
            "selected_binding":selected.to_dict(),"test_measurement_ref":measurement_ref,
            "test_mean_target_nll":measured["mean_target_nll"],"test_exact_correct":measured["exact_correct"],"test_count":len(m["test"]),
            "wall_seconds":(datetime.now(timezone.utc)-datetime.fromisoformat(start["started_at"])).total_seconds(),
            "external_certification":False,"production_promotion":False,"behavioral_gain_claimed":False,
            "lineage":LineageStore(rt.settings.state_root).get_current(lineage)}
        result=store.complete_final(program,identity,result)
        rt.state_repository.assert_integrity()
        return result


def run_benchmark(root):
    root=Path(root).resolve()
    if not (root/"manifest.json").exists():
        prepare(root)
    _,_,_,plan=load(root)
    # Fixed interleaving; shared model/data and matched training seeds.
    results=[run_program(root,policy,seed) for seed in plan.seeds for policy in plan.policies]
    summary={"schema":"public-cpu-autonomy-comparison.v1","programs":results,
        "all_cycles_completed":all(len(r["cycles"])==3 for r in results),
        "total_actual_optimizer_updates":sum(r["actual_optimizer_updates"] for r in results),
        "policy_analysis":compare_policies(results,plan),"adaptive_superiority_established":False,"external_certification":False}
    (root/"comparison.json").write_bytes(canonical_bytes(summary))
    return summary


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("run","program","cold-program","cold"))
    parser.add_argument("root",type=Path)
    parser.add_argument("--policy",choices=("fixed","heuristic","adaptive"),default="adaptive")
    parser.add_argument("--seed",type=int,default=1117)
    args=parser.parse_args()
    if args.action=="program" and not (args.root/"manifest.json").exists(): prepare(args.root)
    if args.action in {"cold","cold-program"}:
        from hephaestus.evaluation.cpu_autonomy_verification import verify_program
        if args.action=="cold-program":
            outcome=verify_program(args.root,f"cpu-{args.policy}-{args.seed}")
        else:
            rt,m,anchor,plan=load(args.root,read_only=True)
            checks=[verify_program(args.root,f"cpu-{policy}-{seed}") for seed in plan.seeds for policy in plan.policies]
            store=CpuAutonomyStore(rt.state_repository,rt.artifact_store)
            results=[read_verified_json(rt.artifact_store,store.final(c["program_id"])["result_ref"]) for c in checks]
            published=json.loads((args.root/"comparison.json").read_text())
            if published["policy_analysis"]!=compare_policies(results,plan):
                raise ValueError("cold comparison statistics differ")
            outcome={"programs":checks,"policy_analysis_recomputed":True,"external_certification":False}
    else:
        outcome=run_benchmark(args.root) if args.action=="run" else run_program(args.root,args.policy,args.seed)
    print(json.dumps(outcome,indent=2))
