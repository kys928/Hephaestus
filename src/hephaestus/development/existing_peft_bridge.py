"""One genuine PEFT CPU *development* run through existing Hephaestus adapters.

Original locally generated math strings only; never a claim of production
dataset approval, five-role completion, or certified evaluation. The guarded
production 45-stage composition and its CUDA-only role policy are untouched.
"""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.control.staged_state import StagedAutonomousServices
from hephaestus.development.cpu_cycle import (
    SEED, source_data, isolate_data, plan_evaluation, canonical, digest,
    save_json, EVAL_TASK_FIELDS, FROZEN_PACK_NAME,
)
from hephaestus.evaluation.pack_loader import load_eval_pack
from hephaestus.generation.backends import TransformersCausalLMGenerationBackend
from hephaestus.generation.models import GenerationTask
from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.schemas.experiment_contract import ExperimentProposal
from hephaestus.schemas.trainable_data_contract import TrainableDataContract
from hephaestus.scoring.behavioral import evaluate_behavioral_sample
from hephaestus.training.hf_lifecycle import (
    TransformersTrainingLifecycleService, validate_checkpoint_manifest,
)
from hephaestus.training.peft_support import validate_recipe


def _hash(path: Path) -> str:
    return digest(path.read_bytes())


def _lexicon():
    from tokenizers import pre_tokenizers
    splitter = pre_tokenizers.Whitespace()
    # Author-defined task grammar and mathematical numeral alphabet, not
    # harvested training/heldout strings or external dataset material.
    corpus = [
        "<|prompt|> Return JSON with integer sum for 1 plus 2. <|target|> {\"sum\":3}",
        *[str(i) for i in range(19)],
    ]
    tokens = sorted({segment for text in corpus for segment, _ in splitter.pre_tokenize_str(text)})
    specials = ["<pad>", "<eos>", "<unk>", "<bos>", "<|prompt|>", "<|target|>"]
    return {token:i for i,token in enumerate([*specials,*[x for x in tokens if x not in specials]])}


def _local_model(root: Path) -> tuple[Path, str, int, Any]:
    import torch
    import transformers
    import tokenizers
    modeldir = root / "local-mini-gpt2"
    modeldir.mkdir(parents=True)
    vocab = _lexicon()
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
        bos_token="<bos>", unk_token="<unk>",
        additional_special_tokens=["<|prompt|>", "<|target|>"],
    )
    torch.manual_seed(SEED)
    model = transformers.AutoModelForCausalLM.from_config(transformers.GPT2Config(
        vocab_size=len(tokenizer), n_positions=128, n_ctx=128,
        n_embd=32, n_layer=1, n_head=2,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    ))
    # Avoid sampling only nonprinting special tokens from the locally
    # initialized base. This is a symmetric, saved generation policy used by
    # BOTH baseline and trained candidate, not synthetic text substitution.
    # Transformers v5 renamed the additional special-token interface.
    # Resolve the two explicitly authored markers by exact token identity,
    # rather than silently omitting them from the generation suppression.
    authored_marker_ids = [
        tokenizer.convert_tokens_to_ids(marker)
        for marker in ("<|prompt|>", "<|target|>")
    ]
    if (len(set(authored_marker_ids)) != 2
            or any(type(token_id) is not int or token_id == tokenizer.unk_token_id
                   for token_id in authored_marker_ids)):
        raise ValueError("local development marker IDs are not independently resolved")
    model.generation_config.suppress_tokens = [
        tokenizer.pad_token_id, tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        *authored_marker_ids,
    ]
    model.generation_config.min_new_tokens = 1
    model.save_pretrained(modeldir, safe_serialization=True)
    tokenizer.save_pretrained(modeldir)
    return modeldir, directory_content_identity(modeldir), sum(p.numel() for p in model.parameters()), tokenizer


def _preparation(root: Path, train: list[dict], tokenizer, modeldir: Path) -> dict:
    # Local development records are NOT the production approved-source
    # acquisition receipt. They are never routed through the production
    # selection/finalization gate.
    source = root / "trainable.jsonl"
    from hephaestus.schemas.prompt_target_supervision import PromptTargetSupervision
    records = [{
        "text": "<|prompt|> " + row["prompt"] + " <|target|> " + row["response"],
        "record_kind": "prompt_target", "source_id": row["id"],
        "supervision": PromptTargetSupervision(row["prompt"], row["response"],
            "<|prompt|> {prompt} <|target|> {target}").to_dict(),
    } for row in train]
    source.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in records), encoding="utf-8")
    source_sha = _hash(source)
    unknown = 0
    tokens = 0
    for row in records:
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        unknown += ids.count(tokenizer.unk_token_id)
        tokens += len(ids)
    if not tokens or 100*unknown > tokens:
        raise ValueError("source fails 1% original-style tokenizer unknown ceiling")
    contract = TrainableDataContract(
        contract_id="original-development-math-v1", run_id="cpu-dev-peft",
        manifest_id="original-development-math-manifest-v1",
        processed_dataset_ref=str(source),
        schema_version="trainable-data.v1", min_tokens=1,
    )
    contract_path = root / "trainable_data_contract.json"
    save_json(contract_path, contract.to_dict())
    evidence = {
        "processed_dataset_ref":str(source),
        "processed_content_hash":source_sha,
        "source_scope":"original_generated_local_development_only",
        "production_approval_ref":None,
        "rights_basis":"Original deterministic arithmetic facts and authored instruction template",
        "wrapper":{"kind":"explicit_prompt_target_template",
                   "template":"<|prompt|> {prompt} <|target|> {target}"},
        "prompt_target_boundary":{
            "status":"explicit","prompt_marker":"<|prompt|>","target_marker":"<|target|>"},
        "tokenizer_compatibility":{
            "status":"checked","tokenizer_ref":str(modeldir),
            "checker_id":"development-actual-tokenizer-unknown-count-v1",
            "compatible":True,"unknown_token_count":unknown,
            "total_tokens":tokens,"max_unknown_fraction":0.01,
        },
    }
    evidence_path = root / "processing_evidence.json"
    save_json(evidence_path,evidence)
    return {
        "processed_dataset_ref":str(source),"processed_dataset_hash":source_sha,
        "trainable_data_contract_ref":str(contract_path),
        "trainable_data_contract_hash":_hash(contract_path),
        "processing_evidence_ref":str(evidence_path),
        "processing_evidence_hash":_hash(evidence_path),
    }


def _baseline_checkpoint(root: Path, modeldir: Path) -> tuple[Path,dict]:
    import shutil
    checkpoint = root / "baseline-checkpoint"
    (checkpoint / "model").mkdir(parents=True)
    shutil.copytree(modeldir, checkpoint / "model", dirs_exist_ok=True)
    shutil.copytree(modeldir, checkpoint / "tokenizer")
    components={
        p.relative_to(checkpoint).as_posix():_hash(p)
        for p in sorted(checkpoint.rglob("*")) if p.is_file()
    }
    # Exact existing checkpoint verifier: no bypass of model/tokenizer bytes.
    manifest={
        "schema":"checkpoint-manifest.v1","partial_write":False,
        "components":components,"manifest_hash":digest(canonical(components)),
    }
    save_json(checkpoint / "checkpoint_manifest.json",manifest)
    if not validate_checkpoint_manifest(checkpoint)[0]:
        raise ValueError("original baseline checkpoint failed verification")
    return checkpoint, {
        "backend":"transformers_causal_lm",
        "training_mode":"full_finetune",
        "model_artifact_ref":str(checkpoint/"model"),
        "tokenizer_artifact_ref":str(checkpoint/"tokenizer"),
        "trust_remote_code":False,
    }


def _score(backend, loading:dict, tasks:list[dict], *, run_id:str) -> dict:
    prompts=[GenerationTask(t["task_id"],"structure_tests", "<|prompt|> "+t["prompt"]+" <|target|> ",SEED)
             for t in tasks]
    generations=backend.generate_batch(prompts,run_id=run_id,loading_instructions=loading,
                                       decoding_config={"max_new_tokens":18,"temperature":0.0,"top_p":1.0})
    if len(generations)!=len(tasks):
        raise ValueError("missing real generator outputs")
    samples=[]
    for task,generated in zip(tasks,generations,strict=True):
        output=generated.output.strip()
        score=evaluate_behavioral_sample(task,output,SEED)
        samples.append({"task_id":task["task_id"],"output":output,
                        "output_sha256":digest(output.encode()),"checks":score.to_dict(),
                        "finish_reason":generated.finish_reason})
    return {"samples":samples,
            "mean_score":sum(s["checks"]["overall_score"] for s in samples)/len(samples),
            "hard_pass_count":sum(s["checks"]["deterministic_passed"] for s in samples)}


def _frozen_tasks(pack:dict)->list[dict]:
    tasks=[]
    for key in EVAL_TASK_FIELDS:
        tasks.extend(pack["eval_pack"].get(key,[]))
    return tasks


def _checks(run:dict,name:str)->int:
    return sum(check["passed"] for sample in run["samples"]
               for check in sample["checks"]["checks"] if check["name"]==name)


def prepare_experiment(root:Path, *,steps:int) -> dict:
    """Original-development input cannot mint a production permission."""
    if not 1<=steps<=12:
        raise ValueError("PEFT CPU development steps must be 1..12")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("existing experimental run may not be overwritten")
    root.mkdir(parents=True,exist_ok=True)
    all_rows=source_data()
    train,heldout,contamination=isolate_data(all_rows)
    frozen=load_eval_pack(FROZEN_PACK_NAME)
    plan=plan_evaluation(all_rows,heldout,frozen)
    plan["backend"]="existing_transformers_lifecycle_lora_cpu"
    plan["base_is_random_initialized"]=True
    plan["multimodel_frozen_cuda_roles_executed"]=False
    plan_sha=save_json(root/"evaluation_plan.json",plan)
    modeldir,rev,parameter_count,tok=_local_model(root)
    data=_preparation(root,train,tok,modeldir)
    recipe={"rank":2,"alpha":4.0,"dropout":0.0,"target_modules":["c_attn"]}
    validate_recipe(recipe,model_id=str(modeldir),revision=rev,mode="lora_finetune")
    constraints={
        "backend_id":"transformers_causal_lm",
        "model_id":str(modeldir),"model_revision":rev,
        "architecture_family":"gpt2",
        "tokenizer_id":str(modeldir),"tokenizer_revision":rev,"tokenizer_ref":str(modeldir),
        "training_mode":"lora_finetune","peft_config":recipe,
        **data,
        "optimizer":"adamw","scheduler":"constant","parameter_count":parameter_count,
        "hidden_size":32,"vocabulary_size":len(tok),
        "special_token_ids":{"pad_token_id":tok.pad_token_id,
                            "eos_token_id":tok.eos_token_id,
                            "unk_token_id":tok.unk_token_id,
                            "bos_token_id":tok.bos_token_id},
        "context_length":96,"device":"cpu","dtype":"float32",
        "seed":SEED,"learning_rate":0.009,"batch_size":4,
        "gradient_accumulation_steps":1,"max_steps":steps,
        "checkpoint_every_steps":steps,"logging_every_steps":1,
        "max_total_tokens":10000,"local_files_only":True,
        "prompt_masking":"mask_prompt_for_prompt_target",
    }
    proposal=ExperimentProposal(
        experiment_id="cpu-peft-development-experiment",run_id="cpu-peft-development",
        lineage_id="local-development-only",stage_name="smoke_test",
        diagnosis_report_id="dev-input-schema-check",
        intervention_id="dev-original-math-peft",
        primary_variable="training_recipe",training_constraints=constraints,
        status="ready",required_approvals=[],
        metadata={
            "source_authority":"repository-original-development-data",
            "production_staged_acquisition_approval":False,
            "pretrained_frozen_five_role_execution":False,
        },
    )
    save_json(root/"local_development_proposal.json",proposal.to_dict())
    _,baseline_loading=_baseline_checkpoint(root,modeldir)
    # The exact existing inference backend is used for both baseline and
    # post-PEFT candidate. No fake generation worker or changed eval plan.
    return {
        "proposal":proposal,"plan":plan,"plan_sha":plan_sha,"frozen":frozen,
        "baseline_loading":baseline_loading,"contamination":contamination,
    }


def run_existing_peft_cycle(root:Path, *,steps:int=8,timeout_seconds:int=180)->dict:
    data=prepare_experiment(root,steps=steps)
    plan=data["plan"]
    backend=TransformersCausalLMGenerationBackend(device="cpu",dtype="float32",batch_size=4)
    frozen_tasks=_frozen_tasks(data["frozen"])
    baseline=_score(backend,data["baseline_loading"],plan["tasks"],run_id="dev-peft-baseline")
    baseline_reg=_score(backend,data["baseline_loading"],frozen_tasks,run_id="dev-peft-base-frozen")
    before_ref=save_json(root/"baseline_generation.json",{"target":baseline,"regression":baseline_reg,
                                                           "plan_ref":data["plan_sha"]})
    service=TransformersTrainingLifecycleService(root/"lifecycle",maximum_allowed_steps=12)
    started=service.launch(data["proposal"])
    if started.status not in {"running","queued","preparing"}:
        raise RuntimeError(f"PEFT service refused development job: {[i.code for i in started.issues]}")
    deadline=time.monotonic()+timeout_seconds
    state=service.status(data["proposal"].run_id)
    while state.status not in {"completed","failed","cancelled"} and time.monotonic()<deadline:
        time.sleep(0.12)
        state=service.status(data["proposal"].run_id)
    if state.status!="completed":
        raise RuntimeError("real PEFT job failed or exceeded bounded CPU time: "+state.status
                           +" "+",".join(i.code for i in state.issues))
    checkpoint=Path(state.checkpoint_refs[-1])
    ok,manifest_hash=validate_checkpoint_manifest(checkpoint)
    if not ok or not (checkpoint/"model"/"adapter_model.safetensors").is_file():
        raise RuntimeError("actual PEFT checkpoint/LoRA adapter bytes not verified")
    from safetensors import safe_open
    with safe_open(str(checkpoint/"model"/"adapter_model.safetensors"),framework="pt",device="cpu") as f:
        trained_weights={key:f.get_tensor(key) for key in f.keys()}
    if not trained_weights or not any("lora_B" in k and (v!=0).any().item() for k,v in trained_weights.items()):
        raise ArithmeticError("no learned LoRA weights recorded")
    loading=json.loads((checkpoint/"loading_instructions.json").read_text())
    candidate=_score(backend,loading,plan["tasks"],run_id="dev-peft-candidate")
    candidate_reg=_score(backend,loading,frozen_tasks,run_id="dev-peft-candidate-frozen")
    after_ref=save_json(root/"candidate_generation.json",{"target":candidate,"regression":candidate_reg,
                                                          "plan_ref":data["plan_sha"],
                                                          "checkpoint_manifest_hash":manifest_hash})
    exact_before=_checks(baseline,"correct_sum")
    exact_after=_checks(candidate,"correct_sum")
    json_before=_checks(baseline,"valid_typed_json")
    json_after=_checks(candidate,"valid_typed_json")
    previous={(s["task_id"],c["name"]):c for s in baseline_reg["samples"] for c in s["checks"]["checks"]}
    following={(s["task_id"],c["name"]):c for s in candidate_reg["samples"] for c in s["checks"]["checks"]}
    if previous.keys()!=following.keys():
        raise ValueError("frozen regression task identities differ")
    new_fails=[str(k) for k,old in previous.items() if old["hard"] and old["passed"]
               and not following[k]["passed"]]
    action=JudgePolicy().decide_exit_action(
        deterministic_passed=not new_fails,confidence=0.0,monitor_outcome="healthy",
        promotion_state="rejected" if new_fails else "inconclusive",
        has_candidate_checkpoint=True)
    if action.value not in {"reject_checkpoint","continue_lineage_best"}:
        raise PermissionError("development cannot promote weights")
    verdict={
        "schema":"hephaestus-development-real-peft-judge.v1",
        "primary_metric":"correct_sum",
        "primary_before":exact_before,"primary_after":exact_after,
        "json_before":json_before,"json_after":json_after,
        "new_hard_regression_failures":new_fails,
        "substantive_target_improvement":exact_after>exact_before and not new_fails,
        "judge_policy_action":action.value,
        "suggested_next_experiment":"address_arithmetic_generalization" if exact_after<=exact_before
                                    else "repeat_and_validate",
        "scientific_certified":False,"promoted":False,"45_stage_campaign_advanced":False,
        "independent_holdout":False,"full_repeatability_proven":False,
    }
    judge_ref=save_json(root/"judge_verdict.json",verdict)
    report={
        "schema":"hephaestus-existing-peft-dev-bridge.v1",
        "status":"completed_noncertified_development",
        "no_external_dataset_access":True,"source":"original generated arithmetic",
        "approval":"no_production_approval_claimed",
        "precommitted_eval_plan":data["plan_sha"],
        "same_original_frozen_pack_hash":data["frozen"]["content_hash"],
        "same_task_plan_baseline_and_candidate":True,
        "train_dataset_hash":data["proposal"].training_constraints["processed_dataset_hash"],
        "model_base_hash":data["proposal"].training_constraints["model_revision"],
        "actual_training_adapter":"TransformersTrainingLifecycleService",
        "actual_generation_adapter":"TransformersCausalLMGenerationBackend",
        "actual_scoring_adapter":"evaluate_behavioral_sample",
        "actual_judge_adapter":"JudgePolicy",
        "actual_peft_checkpoint_manifest":manifest_hash,
        "peft_weight_names":sorted(trained_weights),
        "training_handle":state.to_dict(),
        "baseline_ref":before_ref,"candidate_ref":after_ref,"judge_ref":judge_ref,
        "training_steps":steps,"original_governed_45_stage_progress":"11/45_unchanged",
        "frozen_five_role_production_models_not_loaded":True,
        "multimodel_cpu_inference_proven":False,
        "formal_scientific_certification":False,
    }
    ref=save_json(root/"peft_cycle_manifest.json",report)
    verified=verify_peft_cycle(root)
    save_json(root/"peft_replay.json",verified)
    return {"status":report["status"],"manifest_ref":ref,"verdict":verdict,
            "checkpoint_manifest":manifest_hash,"training_steps":steps,
            "replay":verified}


def verify_peft_cycle(root:Path)->dict:
    report=json.loads((root/"peft_cycle_manifest.json").read_text())
    plan_ref=_hash(root/"evaluation_plan.json")
    if report["precommitted_eval_plan"]!=plan_ref or report["original_governed_45_stage_progress"]!="11/45_unchanged":
        raise ValueError("experiment mutated frozen development boundaries")
    if report["approval"]!="no_production_approval_claimed" or not report["frozen_five_role_production_models_not_loaded"]:
        raise ValueError("development metadata claims unauthorized approval or model role")
    for filename,key in (("baseline_generation.json","baseline_ref"),
                         ("candidate_generation.json","candidate_ref"),
                         ("judge_verdict.json","judge_ref")):
        if _hash(root/filename)!=report[key]:
            raise ValueError("experiment replay output modified: "+filename)
    records=json.loads((root/"candidate_generation.json").read_text())
    checkpoint=Path(report["training_handle"]["checkpoint_refs"][-1])
    ok,mh=validate_checkpoint_manifest(checkpoint)
    if not ok or mh!=report["actual_peft_checkpoint_manifest"] or records["checkpoint_manifest_hash"]!=mh:
        raise ValueError("real LoRA checkpoint changed or cannot replay")
    for filename in ("baseline_generation.json","candidate_generation.json"):
        packet=json.loads((root/filename).read_text())
        if packet["plan_ref"]!=plan_ref:
            raise ValueError("baseline and candidate used different frozen task plans")
    return {"schema":"hephaestus-existing-peft-dev-replay.v1","valid":True,
            "real_checkpoint_manifest_verified":True,"original_campaign_advanced":False,
            "production_approvals_unaffected":True,"frozen_role_claim":False}
