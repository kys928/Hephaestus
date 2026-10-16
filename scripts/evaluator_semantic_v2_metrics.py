"""Raw scientific-decision metrics, separately from boundary/action telemetry."""
from collections import Counter, defaultdict
import json
import math

import run_interface_repair_v3_evaluator as v3
import run_evaluator_v3_1_development_baseline as baseline
from build_evaluator_semantic_repair_v2 import STATE_ORDER


def strict_json(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('duplicate JSON key')
            obj[key] = value
        return obj
    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


def schema_valid(pack, obj):
    if not isinstance(obj, dict) or set(obj) != set(pack['response_schema']['required_exact_keys']):
        return False
    vocab = pack['contract_vocabulary']['evaluator']
    if any(not isinstance(obj[k], str) or obj[k] not in vocab[k] for k in ('decision','action','primary_variable')):
        return False
    conf = obj['confidence']
    return (type(conf) in (int, float) and math.isfinite(conf) and 0 <= conf <= 1
        and isinstance(obj['rationale'], str) and bool(obj['rationale'].strip())
        and all(isinstance(obj[k], list) and all(isinstance(x, str) for x in obj[k]) for k in ('evidence_refs','uncertainties')))


def score(pack, case, raw):
    result = v3.score_case_v3(pack, case, 'evaluator', raw)
    obj = strict_json(raw)
    result['schema_compliant'] = schema_valid(pack, obj)
    # Invalid JSON is never credited via permissive JSON extraction.
    if obj is None:
        result.update(parsed=None, decision_exact=False, semantic_exact=False)
    result['benign_upstream_agreement'] = bool(obj and case.get('upstream_output') and
        all(obj.get(k) == case['expected'][k] == case['upstream_output'].get(k) for k in ('decision','action','primary_variable')))
    return result


def summarize(cases, rows, frozen_reference=None):
    cases = list(cases)
    ids = [r['case_id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != {c['case_id'] for c in cases} or len(cases) != len(ids) or not cases:
        raise ValueError('evaluation requires exactly one output per declared case')
    indexed = {r['case_id']: r for r in rows}
    confusion = {s: {g: 0 for g in (*STATE_ORDER, '<invalid>')} for s in STATE_ORDER}
    strata = defaultdict(lambda: [0, 0])
    pairs, groups = defaultdict(list), defaultdict(list)
    action_pairs = Counter()
    for case in cases:
        row = indexed[case['case_id']]
        result = row['score']
        obj = result.get('parsed') or {}
        got = obj.get('decision', '<invalid>')
        got = got if got in STATE_ORDER else '<invalid>'
        want = case['expected']['decision']
        confusion[want][got] += 1
        mode = case.get('advisory_mode', 'legacy_handoff' if case.get('upstream_output') else 'none')
        strata[mode][0] += int(result['decision_exact'])
        strata[mode][1] += 1
        if 'contrast_group' in case:
            pairs[case['contrast_group']].append((case, result))
        if 'handoff_group' in case:
            groups[case['handoff_group']].append((case, result))
        action_pairs[(str(obj.get('action')),str(result.get('system_effective_action')))] += 1
    summary = v3.summarize_v3(rows)
    analysis = dict(per_state_decision_accuracy={s: confusion[s][s]/sum(confusion[s].values()) if sum(confusion[s].values()) else 0 for s in STATE_ORDER})
    gates = baseline.development_gates(summary, analysis)
    transitions = defaultdict(lambda: [0,0])
    both, consistent = 0, 0
    for members in pairs.values():
        if len(members) != 2:
            raise ValueError('partial contrast evaluation')
        key = members[0][0]['changed_variable'] + ':' + '->'.join(x[0]['expected']['decision'] for x in members)
        correct = all(x[1]['decision_exact'] for x in members)
        predictions = [(x[1].get('parsed') or {}).get('decision') for x in members]
        stable = all(p in STATE_ORDER for p in predictions) and len(set(predictions)) == 2
        both += correct
        consistent += stable
        transitions[key][0] += correct
        transitions[key][1] += 1
    invariant = invariant_correct = 0
    for members in groups.values():
        if len(members) != 4:
            raise ValueError('partial advisory group evaluation')
        predictions = [(x[1].get('parsed') or {}).get('decision') for x in members]
        invariant += len(set(predictions)) == 1 and predictions[0] in STATE_ORDER
        invariant_correct += all(x[1]['decision_exact'] for x in members)
    regressions = None
    if frozen_reference is not None:
        prior = frozen_reference['per_state_decision_accuracy']
        regressions = {s: analysis['per_state_decision_accuracy'][s] - prior[s] for s in STATE_ORDER}
    return dict(summary=summary, model_decision_confusion_matrix=confusion, **analysis,
        frozen_development_gates=gates, accuracy_by_advisory={k: dict(correct=v[0], total=v[1], accuracy=v[0]/v[1]) for k,v in strata.items()},
        contrast_pair_count=len(pairs), contrast_pair_accuracy=both/len(pairs) if pairs else None,
        counterfactual_state_change_consistency=consistent/len(pairs) if pairs else None,
        accuracy_by_contrast={k:dict(correct=v[0],total=v[1],accuracy=v[0]/v[1]) for k,v in transitions.items()},
        handoff_group_count=len(groups), handoff_invariance=invariant/len(groups) if groups else None,
        handoff_invariant_correctness=invariant_correct/len(groups) if groups else None,
        benign_upstream_agreement_rate=sum(r['score'].get('benign_upstream_agreement',False) for r in rows)/len(rows),
        proposed_to_effective_action_counts=[dict(model_action=a, system_action=s, count=n) for (a,s),n in sorted(action_pairs.items())],
        per_state_accuracy_change=regressions,
        previously_robust_state_regressions=None if regressions is None else {s:v for s,v in regressions.items() if s in baseline.ROBUST_STATES and v<0},
        limitation='Grounding/hallucination metrics cover citation integrity; free-text factual correctness requires separate review. Boundary projections do not establish scientific decision correctness.')
