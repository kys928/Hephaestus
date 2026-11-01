"""Actual model generation and teacher-forced target NLL, without changing weights."""
from hephaestus.evaluation.prospective_pipeline import generate_cpu, verify_model_binding
from hephaestus.evaluation.exact_json import score_prediction
from hephaestus.training.cpu_chat_json_recipe import chat_prompt


def measure(binding, rows, design):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    generated = generate_cpu(binding, rows, design)
    tokenizer = AutoTokenizer.from_pretrained(binding.tokenizer_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(binding.model_path, local_files_only=True,
                                               trust_remote_code=False, dtype=torch.float32).to("cpu")
    if binding.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model,binding.adapter_path,is_trainable=False,local_files_only=True)
    model.eval()
    losses = []
    for row in rows:
        prefix = tokenizer.encode(chat_prompt(tokenizer,row["prompt"]),add_special_tokens=False)
        target = tokenizer.encode(row["target"]+tokenizer.eos_token,add_special_tokens=False)
        if len(prefix)+len(target)>model.config.max_position_embeddings or not target:
            raise ValueError("CPU NLL evaluation cannot truncate prompt/target")
        ids = torch.tensor([prefix+target],dtype=torch.long)
        labels = ids.clone(); labels[:,:len(prefix)] = -100
        with torch.inference_mode():
            loss = model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels).loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite CPU evaluation NLL")
        losses.append(float(loss))
    verify_model_binding(binding,design)
    scores = [score_prediction(s["answer_text"],r["target"]) for s,r in zip(generated["samples"],rows,strict=True)]
    return {"schema":"cpu-autonomy-measurement.v1","generation":generated,"per_case_target_nll":losses,
        "mean_target_nll":sum(losses)/len(losses),"exact_correct":sum(s["exact_correct"] for s in scores),
        "valid_json":sum(s["valid_json"] for s in scores),"count":len(rows),
        "eos_failures":sum(not s["eos"] for s in generated["samples"]),
        "truncations":sum(s["truncated"] for s in generated["samples"]),
        "nll_scope":"teacher_forced_mean_per_example_target_token_nll_not_exact_behavioral_success"}


def compare_policies(results,plan):
    from itertools import product
    from math import comb
    comparisons={}
    by_key={(r["policy"],r["seed"]):r for r in results}
    if len(by_key)!=9 or set(by_key)!={(p,s) for p in plan.policies for s in plan.seeds}:
        raise ValueError("complete matched nine-program comparison required")
    for other in ("fixed","heuristic"):
        deltas=[by_key[other,s]["test_mean_target_nll"]-by_key["adaptive",s]["test_mean_target_nll"] for s in plan.seeds]
        nonzero=[d for d in deltas if d!=0]
        wins=sum(d>0 for d in nonzero); losses=len(nonzero)-wins
        p=min(1.,2*sum(comb(len(nonzero),k) for k in range(min(wins,losses)+1))/(2**len(nonzero))) if nonzero else 1.
        samples=sorted(sum(draw)/3 for draw in product(deltas,repeat=3))
        comparisons[other]={"paired_test_nll_reductions_favoring_adaptive":deltas,
            "mean_paired_reduction":sum(deltas)/3,"exact_two_sided_training_seed_sign_p":p,
            "enumerated_seed_bootstrap_95_interval":[samples[0],samples[-1]],
            "bootstrap_limit":"27 empirical resamples of three training seeds; weak population uncertainty estimate",
            "maximum_possible_significance_with_three_seeds":.25}
    ordered=sorted(comparisons,key=lambda name:comparisons[name]["exact_two_sided_training_seed_sign_p"])
    prior=0.
    for index,name in enumerate(ordered):
        adjusted=min(1.,max(prior,(2-index)*comparisons[name]["exact_two_sided_training_seed_sign_p"]))
        comparisons[name]["holm_adjusted_p"]=adjusted; prior=adjusted
    return {"comparisons":comparisons,"adaptive_superiority_established":False,
        "reason":"Three matched training seeds cannot establish superiority at alpha .05 using the predeclared sign test.",
        "scope":"Fixed random base, fixed public task sample and finite deterministic policy; no specialist-model contribution claim."}
