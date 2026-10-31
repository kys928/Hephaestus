#!/usr/bin/env python3
"""First one-shot untouched generated-evaluation comparison for a frozen CPU adapter.

No training, no model selection after evaluation, no original five-role CUDA/GPU,
no independent historical certification, no raw evaluation text in report.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

from hephaestus.data.generated_factory import generate_original_splits, generator_code_sha256
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.data.generated_validation import validate_original_example
from hephaestus.storage.verified import canonical_bytes
from hephaestus.llm.cpu_causal_research import MODEL_ID,MODEL_REVISION

SCHEMA="cpu-original-confirmatory-json.v1"
CONFIG=Path("configs/research/cpu_original_json_confirmatory_v1.json")


def validated_config(path):
    value=json.loads(Path(path).read_bytes())
    frozen=json.loads(CONFIG.read_bytes())
    if value!=frozen or set(value)!=set(frozen) or value.get("schema")!=SCHEMA or not (
            value.get("model_promoted") is False and
            value.get("original_five_cuda_gpu_executed") is False and
            value.get("external_certification") is False and
            value.get("model_id")==MODEL_ID and value.get("model_revision")==MODEL_REVISION):
        raise PermissionError("confirmatory protocol must equal precommitted immutable decision")
    return value


def verified_adapter(archive_dir, protocol):
    root=Path(archive_dir)
    manifest=json.loads((root/"archive_sha256.json").read_bytes())
    files={p.relative_to(root).as_posix():p for p in root.rglob("*")
           if p.is_file() and not p.is_symlink() and p.name!="archive_sha256.json"}
    if not isinstance(manifest,dict) or set(files)!=set(manifest):
        raise PermissionError("original CPU artifact file inventory mismatch")
    for key,path in files.items():
        if sha256(path.read_bytes()).hexdigest()!=manifest[key]:
            raise PermissionError("original CPU artifact SHA256 mismatch")
    result=json.loads((root/"report.json").read_bytes())
    if (result.get("schema")!="cpu-chat-json-repair-result.v1" or
            result.get("model_revision")!=protocol["model_revision"] or
            result.get("model_weights_sha256")!=protocol["model_weights_sha256"] or
            result.get("tokenizer_sha256")!=protocol["tokenizer_sha256"] or
            result.get("adapter_sha256")!=protocol["adapter_sha256"] or
            result.get("training_seed")!=protocol["train_seed"] or
            result.get("optimizer_steps")!=96 or
            result.get("frozen_development_ref")!=protocol["development_id"] or
            result.get("evaluation_holdout_used") is not False or
            result.get("model_promoted") is not False or
            "adapter/adapter_model.safetensors" not in manifest or
            protocol["adapter_sha256"]!="sha256:"+manifest["adapter/adapter_model.safetensors"]):
        raise PermissionError("selected frozen original CPU candidate not authenticated")
    return root/"adapter"


def chat_prompt(tokenizer, task):
    if not tokenizer.chat_template:
        raise ValueError("pinned tokenizer chat template required")
    return tokenizer.apply_chat_template([
        {"role":"system","content":"Return only one valid JSON object with exactly the requested answer, with no markdown, code, or prose."},
        {"role":"user","content":task}
    ],tokenize=False,add_generation_prompt=True)


def check_output(text, target):
    try:
        a,b=json.loads(text),json.loads(target)
    except (ValueError,TypeError):
        return False,False
    valid=isinstance(a,dict)
    return valid,valid and a==b and all(type(a[k]) is type(v) for k,v in b.items())


def generate(model,tokenizer,examples,maximum_tokens):
    import torch
    model.eval()
    totals=Counter()
    family={}
    rows=[]
    for example in examples:
        prompt=chat_prompt(tokenizer,example["prompt"])
        tokens=tokenizer(prompt,return_tensors="pt",add_special_tokens=False)
        if tokens.input_ids.shape[1]+maximum_tokens>model.config.max_position_embeddings:
            raise ValueError("frozen evaluation prompt cannot truncate")
        with torch.inference_mode():
            result=model.generate(**tokens,do_sample=False,max_new_tokens=maximum_tokens,
                    pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
        output=tokenizer.decode(result[0,tokens.input_ids.shape[1]:],
                                skip_special_tokens=True).strip()
        valid,correct=check_output(output,example["target"])
        totals["valid_json"]+=int(valid)
        totals["exact_correct"]+=int(correct)
        row=family.setdefault(example["family"],{"count":0,"valid_json":0,"exact_correct":0})
        row["count"]+=1;row["valid_json"]+=int(valid);row["exact_correct"]+=int(correct)
        rows.append({"sample_id":example["sample_id"],
                     "output_sha256":"sha256:"+sha256(output.encode()).hexdigest(),
                     "valid_json":valid,"exact_correct":correct})
    return {"count":len(examples),"counts":dict(totals),"families":family,"samples":rows}


def run(config_path,archive_dir,output,cache_dir):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    protocol=validated_config(config_path)
    if output.exists():
        raise FileExistsError("confirmatory report cannot be replaced")
    adapter=verified_adapter(archive_dir,protocol)
    splits,_=generate_original_splits(GeneratedDataSpec(protocol["generator_seed"]))
    # Evaluation was created and isolated by original source generation, before
    # selecting/training any of these adapters; use it only for this final pass.
    evaluation=splits["evaluation"]
    if len(evaluation)!=protocol["evaluation_sample_count"]:
        raise PermissionError("original frozen evaluation set cardinality changed")
    eval_sha="sha256:"+sha256(canonical_bytes({
        "schema":"original-task-examples.v1","split":"evaluation","examples":evaluation})).hexdigest()
    if generator_code_sha256()!=protocol["generator_sha256"]:
        raise PermissionError("original generator code is not frozen")
    if len({validate_original_example(r) for r in evaluation})!=len(evaluation):
        raise ValueError("invalid frozen confirmatory tasks")
    torch.set_num_threads(2)
    torch.manual_seed(1117)
    snapshot=Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(cache_dir),
        allow_patterns=["config.json","tokenizer.json","tokenizer_config.json",
                        "special_tokens_map.json","model.safetensors","generation_config.json"]))
    if snapshot.name!=MODEL_REVISION or "sha256:"+sha256((snapshot/"model.safetensors").read_bytes()).hexdigest()!=protocol["model_weights_sha256"] or (
            "sha256:"+sha256((snapshot/"tokenizer.json").read_bytes()).hexdigest()!=protocol["tokenizer_sha256"]):
        raise PermissionError("frozen base model or tokenizer bytes changed")
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    started=time.monotonic()
    base=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
                                              trust_remote_code=False,dtype=torch.float32).to("cpu")
    baseline=generate(base,tokenizer,evaluation,protocol["max_new_tokens"])
    del base
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
                                               trust_remote_code=False,dtype=torch.float32).to("cpu")
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True)
    candidate=generate(model,tokenizer,evaluation,protocol["max_new_tokens"])
    if [r["sample_id"] for r in baseline["samples"]] != [r["sample_id"] for r in candidate["samples"]]:
        raise PermissionError("evaluation identities changed between models")
    report={"schema":"cpu-original-generated-confirmatory-result.v1",
       "original_evaluation_ref":eval_sha,
       "precommitted_protocol_sha256":"sha256:"+sha256(canonical_bytes(protocol)).hexdigest(),
       "original_frozen_adapter_sha256":protocol["adapter_sha256"],
       "baseline":baseline,"candidate":candidate,
       "exact_correct_gain":candidate["counts"].get("exact_correct",0)-baseline["counts"].get("exact_correct",0),
       "valid_json_gain":candidate["counts"].get("valid_json",0)-baseline["counts"].get("valid_json",0),
       "no_training_or_model_selection_on_evaluation":True,
       "original_five_cuda_gpu_executed":False,
       "historical_reference_certification":False,"model_promoted":False,
       "source_evaluation_text_published":False}
    output.write_bytes(canonical_bytes(report))
    print(json.dumps({"evaluation_sha":eval_sha,
                      "correct_before":baseline["counts"].get("exact_correct",0),
                      "correct_after":candidate["counts"].get("exact_correct",0),
                      "valid_before":baseline["counts"].get("valid_json",0),
                      "valid_after":candidate["counts"].get("valid_json",0),
                      "gpu_executed":False},sort_keys=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=CONFIG)
    p.add_argument("--cpu-artifact-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cache-dir",type=Path,default=Path("/tmp/hephaestus-model-cache"))
    x=p.parse_args()
    run(x.config,x.cpu_artifact_dir,x.output,x.cache_dir)


if __name__=="__main__":
    main()
