#!/usr/bin/env python3
"""Real frozen Granite tokenizer and tiny Granite forward/loss compatibility on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer, GraniteConfig, GraniteForCausalLM
import build_evaluator_semantic_repair_v2 as b
from evaluator_semantic_v2_supervision import training_example, target_answer
from prepare_evaluator_semantic_repair_v2 import CFG_PATH


def check(tokenizer_path=None):
    cfg=json.loads(CFG_PATH.read_text())
    model_id='ibm-granite/granite-3.3-8b-instruct'
    revision=cfg['required_base_revision']
    source=tokenizer_path or model_id
    kwargs=dict(local_files_only=True) if tokenizer_path else dict(revision=revision)
    tok=AutoTokenizer.from_pretrained(source,trust_remote_code=False,**kwargs)
    # Check all selected targets for every ablation, including longest adversaries.
    counts={}
    legacy_equivalent=0
    import evaluator_semantic_repair_train as old_train
    for arm in 'BCDE':
        p=b.build_pack(arm);selected,_=b.schedule(p,cfg)
        maximum=0
        for row in selected:
            ex=training_example(tok,cfg['roles']['evaluator'],p,row,'evaluator',cfg['training']['max_sequence_length'])
            mask=ex['labels'][0] != -100
            if not torch.equal(ex['labels'][0][mask], ex['input_ids'][0][mask]):
                raise ValueError('supervised token alignment mismatch')
            if arm=='B':
                prior=old_train.v2.training_example(tok,cfg['roles']['evaluator'],p,row,'evaluator',cfg['training']['max_sequence_length'])
                if not torch.equal(ex['input_ids'],prior['input_ids']) or not torch.equal(ex['labels'],prior['labels']):
                    raise ValueError('B differs from effective V1 supervision: investigate before a paid ablation')
                legacy_equivalent+=1
            maximum=max(maximum,ex['nonpad_tokens'])
        counts[arm]=dict(cases=len(selected),maximum_tokens=maximum)
    # CPU smoke tests causal shifting/loss and model API on Granite architecture;
    # this is deliberately not a claim to load the full 8B parent on CPU.
    torch.manual_seed(7)
    tiny=GraniteForCausalLM(GraniteConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
         num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64))
    ids=torch.tensor([[1,2,3,4,5]])
    labels=torch.tensor([[-100,-100,-100,4,5]])
    output=tiny(input_ids=ids,labels=labels)
    expected=torch.nn.functional.cross_entropy(output.logits[:,:-1].reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100)
    torch.testing.assert_close(output.loss,expected)
    output.loss.backward()
    return dict(tokenizer_model_id=model_id,revision=revision,
        tokenizer_backend_sha256=hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest(),
        chat_template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        selected_schedules=counts,legacy_V1_B_token_and_mask_equivalent_cases=legacy_equivalent,
        tiny_granite_forward_loss_backward_passed=True,full_8b_model_loaded=False,
        paid_training_performed=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tokenizer-path');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    report=check(args.tokenizer_path)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,sort_keys=True))
