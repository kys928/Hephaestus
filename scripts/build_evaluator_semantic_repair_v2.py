#!/usr/bin/env python3
"""Controlled B-E ablations; immutable V1 development diagnostic, no sealed data.

B changes supervision only. C replaces V1 contrast rows only. D replaces the
factorial/precedence rows only. E adds advisory matching. Each selected schedule
has 768 examples, balanced states and identical optimizer geometry.
"""
from __future__ import annotations
import copy
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache

import build_evaluator_semantic_invariance_repair_v1 as v1
import evaluator_semantic_repair_common as common
from hephaestus.schemas.evaluator_factors import ScientificFactors, combination_space
from hephaestus.policy.evaluator_semantic_oracle import scientific_state

STATE_ORDER = tuple(v1.STATE_ORDER)
SPACE = combination_space()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def opaque(seed):
    return hashlib.sha256(seed.encode()).hexdigest()[:20].upper()


def distance(a, b):
    return [k for k in a.to_dict() if getattr(a, k) != getattr(b, k)]


@lru_cache(maxsize=None)
def adjacent(left, right, index):
    candidates = [(a, b) for a in SPACE for b in SPACE
                  if scientific_state(a) == left and scientific_state(b) == right
                  and len(distance(a, b)) == 1]
    if not candidates:
        raise ValueError(f'no single-variable transition {left}/{right}')
    return candidates[index % len(candidates)]


def case(factors, family, index, context_seed, *, bank=None):
    state = scientific_state(factors)
    row = v1._make_case(corpus='v2', state=state, latent=factors.to_dict(), family=family,
                       index=index, bank=bank or v1.TRAIN_PHRASES, has_upstream=False,
                       upstream_correct=False, context_seed=context_seed, fact_seed=context_seed)
    row['latent_factors'] = factors.to_dict()
    # Refs and incidental context are independent of the expected label.
    refs = ['E-' + opaque(f'{context_seed}:ref:{i}') for i in range(6)]
    for evidence, ref in zip(row['evidence'], refs):
        evidence['ref'] = ref
    row['allowed_evidence_refs'] = refs
    return row


def contrasts():
    rows = []
    n = 0
    for left, right, count in v1.CONTRAST_EDGES:
        for j in range(count):
            a, b = adjacent(left, right, j)
            group = 'P-' + opaque(f'pair:{n}')
            for side, factors in enumerate((a, b)):
                row = case(factors, 'minimal_contrast', n, group)
                row.update(case_id='C-' + opaque(f'{group}:{side}'), contrast_group=group,
                           changed_variable=distance(a, b)[0], contrast_partner=None)
                # Common advice and refs; exactly one scientific variable changes.
                if n % 2:
                    row.update(kind='interface', upstream_role='planner', upstream_output={
                        'source_role': 'planner', 'authority': 'advisory',
                        'decision': 'candidate_reviewed', 'action': 'hold_candidate',
                        'primary_variable': 'candidate_quality', 'confidence': 0.99})
                rows.append(row)
            n += 1
    return rows


def measurements(f, index):
    """Different thresholds and observations with the same six scientific factors."""
    margin = 0.5 + (index % 19) / 10
    delta = {'positive': 2 * margin, 'neutral': margin / 4, 'negative': -2 * margin}[f.effect]
    limit = 0.05 + (index % 7) / 100
    spread = limit / 2 if f.variance_stable else limit * 2
    return dict(margin=round(margin, 3), delta=round(delta, 3),
                spread=round(spread, 4), limit=round(limit, 4),
                planned=40 + index % 13, present=(40 + index % 13) - (0 if f.complete else 3))


def factorial_row(f, family, index):
    seed = f'{family}:{index}'
    row = case(f, family, index, seed)
    m = measurements(f, index + (10000 if family == 'precedence_and_field_isolation' else 0))
    values = [
        f"Mandatory check failures: {0 if f.gate_pass else 1}. Failure count zero is required.",
        f"Required records: {m['planned']}; persisted records: {m['present']}.",
        'Measured artifact fingerprint ' + ('matches' if f.provenance_valid else 'does not match') + ' the admitted revision.',
        f"Primary quality delta={m['delta']}; symmetric practical margin={m['margin']}. "
        + f"A secondary throughput delta={-m['delta']} is report-only and cannot determine quality.",
        f"Replicate spread={m['spread']}; allowed spread<={m['limit']}.",
        'Required confirmation ' + ('has passed.' if f.recheck_satisfied else 'has not passed.'),
    ]
    # Same evidence semantics in prose, JSON and compact record presentations.
    presentation = index % 3
    if presentation == 1:
        values = [json.dumps({'observation': x}, sort_keys=True) for x in values]
    elif presentation == 2:
        values = ['Observation | ' + x for x in values]
    for ev, value in zip(row['evidence'], values):
        ev['fact'] = value
    row.update(measurements=m, presentation=presentation)
    return row


def factorials():
    rows = []
    for state in STATE_ORDER:
        factors = [f for f in SPACE if scientific_state(f) == state]
        for family, count in [('factorial_generalization', 96), ('precedence_and_field_isolation', 32)]:
            for j in range(count):
                row = factorial_row(factors[j % len(factors)], family, j)
                # Context seeds include state only in a hash, never in visible strings.
                row['case_id'] = 'C-' + opaque(f'{state}:{family}:{j}')
                row['context'] = 'ctx_' + opaque(f'{state}:{family}:{j}:context')
                row['situation'] = f"In {row['context']}, classify the admitted candidate from the verified observations."
                if j % 2:
                    row.update(kind='interface', upstream_role='planner',
                               upstream_output=v1._upstream(row['expected']['action'], f'{state}:{j}', correct=(j // 2) % 2 == 0))
                rows.append(row)
    return rows


def advisory_group(row):
    group = 'G-' + opaque(row['case_id'])
    correct_state = row['expected']['decision']
    # Cross all incorrect states across groups instead of always suggesting a positive verdict.
    wrongs = [s for s in STATE_ORDER if s != correct_state]
    wrong_state = wrongs[int(opaque(group), 16) % len(wrongs)]
    result = []
    for mode in ('none', 'correct', 'incorrect', 'misleading_high_confidence'):
        r = copy.deepcopy(row)
        r.update(case_id='C-' + opaque(f'{group}:{mode}'), handoff_group=group, advisory_mode=mode)
        state = correct_state if mode == 'correct' else wrong_state
        action, primary = v1.ROOT_BY_STATE[state]
        advice = None if mode == 'none' else dict(source_role='planner', authority='advisory',
            decision=state, action=action, primary_variable=primary,
            confidence=0.999 if mode == 'misleading_high_confidence' else 0.6)
        r.update(kind='rehearsal' if advice is None else 'interface', upstream_role=None if advice is None else 'planner', upstream_output=advice)
        result.append(r)
    return result


def build_pack(arm='E'):
    if arm not in 'BCDE' or len(arm) != 1:
        raise ValueError('trainable ablation must be B, C, D or E')
    pack = copy.deepcopy(v1.build_pack())
    rows = pack['partitions']['evaluator']['train']
    if arm != 'B':
        rows = [x for x in rows if x['curriculum_family'] != 'minimal_contrast'] + contrasts()
    if arm in ('D', 'E'):
        rows = [x for x in rows if x['curriculum_family'] == 'minimal_contrast'] + factorials()
    if arm == 'E':
        # Contrast groups remain intact for pair tests. Handoff groups use factorial
        # rows only so advisory matching never changes a scientific contrast.
        rows = [x for x in rows if x['curriculum_family'] == 'minimal_contrast'] + [
            r for x in rows if x['curriculum_family'] != 'minimal_contrast' for r in advisory_group(x)]
    pack['pack_id'] = 'hephaestus-evaluator-semantic-repair-v2-' + arm
    pack['pack_version'] = 2
    pack['ablation'] = arm
    pack['partitions']['evaluator']['train'] = rows
    # Old design claims are superseded, not inherited as validated facts.
    pack['design'] = dict(offline_oracle_only=True, frozen_development=True, fresh_certification_required=True)
    return pack


def schedule(pack, cfg):
    arm = pack['ablation']
    if arm in ('B', 'C', 'D'):
        selected, meta = common.training_schedule(pack, cfg)
        if arm == 'D':
            def difficulty(r):
                f = r['latent_factors']
                return sum(not f[k] for k in f if k != 'effect') + int(f['effect'] == 'negative')
            # Keep the exact chosen dose, progressively increase simultaneous defects.
            by_state = {s: sorted([r for r in selected if r['semantic_root'] == s], key=difficulty) for s in STATE_ORDER}
            selected = [r for block in range(6) for s in STATE_ORDER for r in by_state[s][block*16:(block+1)*16]]
            meta['selected_case_order_sha256'] = digest([r['case_id'] for r in selected])
        return selected, meta
    rows = pack['partitions']['evaluator']['train']
    blocks = [[], [], []]
    if arm == 'D':
        for state in STATE_ORDER:
            for stage, family in enumerate(common.FAMILIES):
                candidates = [r for r in rows if r['semantic_root'] == state and r['curriculum_family'] == family]
                # Minimal contrasts have balanced graph degrees; all 64 per state
                # remain in the full pack. Schedule 32 per state at fixed dose.
                blocks[stage] += common.stable(candidates, f'{cfg["training"]["seed"]}:{stage}')[:32]
    else:
        # Keep all 512 corrected contrasts. Replace the remaining 256 slots with
        # eight matched quartets per state. E must be compared to a matched D
        # control (same selected evidence/order, all advice removed), not ordinary
        # D alone: matching otherwise changes scientific sample multiplicity.
        groups = defaultdict(list)
        for r in rows:
            if 'handoff_group' in r and r['curriculum_family'] == 'factorial_generalization':
                groups[r['handoff_group']].append(r)
        for state in STATE_ORDER:
            candidates = sorted([g for g in groups.values() if g[0]['semantic_root'] == state],
                                key=lambda g: digest(g[0]['handoff_group']))
            blocks[0] += [r for r in rows if r['semantic_root'] == state and r['curriculum_family'] == 'minimal_contrast']
            blocks[1] += [r for g in candidates[:4] for r in g]
            blocks[2] += [r for g in candidates[4:8] for r in g]
    selected = [r for block in blocks for r in block]
    if len(selected) != 768 or len({r['case_id'] for r in selected}) != 768:
        raise ValueError('schedule geometry drift')
    return selected, dict(selected_case_count=len(selected), selected_case_order_sha256=digest([r['case_id'] for r in selected]),
                         selected_state_counts=dict(Counter(r['semantic_root'] for r in selected)),
                         progression=['straightforward', 'counterfactual', 'adversarial'])


def matched_handoff_control(selected):
    result = copy.deepcopy(selected)
    for r in result:
        if 'handoff_group' in r:
            r.update(upstream_output=None, upstream_role=None, kind='rehearsal')
    return result


def validate(pack):
    from validate_evaluator_semantic_repair_v2 import validate_pack
    return validate_pack(pack)


if __name__ == '__main__':
    p = build_pack()
    print(json.dumps(validate(p), sort_keys=True))
