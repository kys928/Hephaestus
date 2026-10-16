"""Explicit state-rationale targets and fail-closed assistant-only token masking."""
import json
import run_interface_repair_v1 as v1


def target_answer(case):
    rationale = case.get('target_rationale')
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError('state-specific target rationale is required')
    e = case['expected']
    obj = dict(decision=e['decision'], action=e['action'], primary_variable=e['primary_variable'],
        confidence=round((float(e['confidence_min']) + float(e['confidence_max']))/2,3),
        evidence_refs=list(case['allowed_evidence_refs']),
        uncertainties=['A verified blocking condition remains unresolved; do not escalate scientific status.']
            if e['decision'] in {'incomplete_evidence','inconclusive','recheck_required'} else [],
        rationale=rationale)
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def training_example(tokenizer, runtime, pack, case, role, maxlen):
    import torch
    if role != 'evaluator' or runtime['load_kind'] != 'causal_lm':
        raise ValueError('V2 initial experiment is frozen Granite causal_lm only')
    question = v1.prompt(pack, case, role)
    answer = target_answer(case)
    messages = [dict(role='user', content=question)]
    prefix = v1.template_ids(tokenizer, runtime['load_kind'], messages, True)
    full = v1.template_ids(tokenizer, runtime['load_kind'], messages + [dict(role='assistant', content=answer)], False)
    # A small LCP can silently supervise prompt/header tokens. Require the whole
    # inference prefix to match and reject tokenization/truncation drift.
    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError('full inference prefix must match the training template')
    if len(full) > maxlen:
        raise ValueError(f'example exceeds max sequence length: {len(full)}>{maxlen}')
    suffix = tokenizer.decode(full[len(prefix):], skip_special_tokens=True)
    if suffix.strip() != answer:
        raise ValueError('assistant token boundary does not serialize the exact target')
    labels = [-100] * len(prefix) + full[len(prefix):]
    return dict(input_ids=torch.tensor([full], dtype=torch.long),
                attention_mask=torch.ones((1, len(full)), dtype=torch.long),
                labels=torch.tensor([labels], dtype=torch.long),
                nonpad_tokens=len(full), supervised_tokens=len(full)-len(prefix),
                token_slots=len(full), target_sha256=__import__('hashlib').sha256(answer.encode()).hexdigest())
