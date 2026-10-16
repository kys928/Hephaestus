"""CPU validation of generated scientific data and lineage, independent of GPUs."""
from collections import Counter, defaultdict
import copy
import json
import re
from pathlib import Path

import build_evaluator_semantic_repair_v2 as b
from hephaestus.schemas.evaluator_factors import ScientificFactors
from hephaestus.policy.evaluator_semantic_oracle import scientific_state


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scientific_signature(r):
    return b.digest({k: r[k] for k in ('evidence', 'allowed_evidence_refs', 'verified_facts', 'situation', 'target_rationale', 'expected')})


def shingles(row):
    text = ' '.join(x['fact'] for x in row['evidence']).lower()
    words = re.findall(r'[a-z]+|\d+(?:\.\d+)?', text)
    return frozenset(zip(words, words[1:], words[2:]))


def validate_pack(pack):
    part = pack['partitions']['evaluator']
    frozen = b.v1.build_pack()['partitions']['evaluator']['development_diagnostic']
    require(part['development_diagnostic'] == frozen, 'immutable development diagnostic changed')
    rows = part['train']
    all_rows = rows + part['preflight'] + frozen
    require(len({r['case_id'] for r in all_rows}) == len(all_rows), 'duplicate case identity')
    require(set(r['semantic_root'] for r in rows) == set(b.STATE_ORDER), 'state coverage incomplete')
    for r in all_rows:
        refs = r['allowed_evidence_refs']
        require(len(refs) == 6 and len(set(refs)) == 6 and [e['ref'] for e in r['evidence']] == refs, 'evidence-reference integrity')
        identifiers = ' '.join([r['case_id'], r['context'], *refs]).lower()
        require(not any(t in identifiers for t in ('improv','equiv','regress','reject','inconcl','recheck','cert','train','dev','split')), 'identifier label leakage')
        if 'latent_factors' in r:
            f = ScientificFactors(**r['latent_factors'])
            require(scientific_state(f) == r['expected']['decision'] == r['semantic_root'], 'oracle target mismatch')
            machine = r['verified_facts']
            require(all(machine[k] == getattr(f, source) for k, source in {
                'deterministic_gate_passed':'gate_pass','evidence_complete':'complete',
                'provenance_valid':'provenance_valid','recheck_satisfied':'recheck_satisfied'}.items()), 'machine-state contradiction')
            if 'measurements' in r:
                m = r['measurements']
                effect = 'positive' if m['delta'] > m['margin'] else 'negative' if m['delta'] < -m['margin'] else 'neutral'
                require(effect == f.effect and (m['spread'] <= m['limit']) == f.variance_stable
                        and (m['present'] == m['planned']) == f.complete, 'numeric-factor contradiction')
    pairs = defaultdict(list)
    groups = defaultdict(list)
    for r in rows:
        if 'contrast_group' in r:
            pairs[r['contrast_group']].append(r)
        if 'handoff_group' in r:
            groups[r['handoff_group']].append(r)
    for members in pairs.values():
        require(len(members) == 2, 'contrast requires two members')
        a, c = members
        diff = b.distance(ScientificFactors(**a['latent_factors']), ScientificFactors(**c['latent_factors']))
        require(diff == [a['changed_variable']] == [c['changed_variable']], 'contrast changes more than declared factor')
        require(a['expected']['decision'] != c['expected']['decision'], 'contrast has no state transition')
        require(all(a[k] == c[k] for k in ('context','situation','allowed_evidence_refs','upstream_output')), 'contrast incidental context changed')
        require(sum(x != y for x, y in zip(a['evidence'], c['evidence'])) == 1, 'contrast must change exactly one observation')
    for members in groups.values():
        require(len(members) == 4 and {r['advisory_mode'] for r in members} == {'none','correct','incorrect','misleading_high_confidence'}, 'incomplete handoff quartet')
        require(len({scientific_signature(r) for r in members}) == 1, 'handoff evidence/target changed')
        for r in members:
            advice = r['upstream_output']
            if r['advisory_mode'] == 'none':
                require(advice is None, 'direct case has advice')
            else:
                require((advice['decision'] == r['expected']['decision']) == (r['advisory_mode'] == 'correct'), 'advisory correctness mismatch')
    arm = pack['ablation']
    if arm in ('D','E'):
        covered = {b.digest(r['latent_factors']) for r in rows if r['curriculum_family'] == 'factorial_generalization'}
        require(covered == {b.digest(f.to_dict()) for f in b.SPACE}, 'incomplete factorial cross product')
    # Detect contamination even after identifiers and whitespace are removed.
    train_facts = {tuple(e['fact'] for e in r['evidence']) for r in rows}
    dev_facts = {tuple(e['fact'] for e in r['evidence']) for r in frozen}
    require(not train_facts & dev_facts, 'exact train/development duplicate')
    train_shingles = set(shingles(r) for r in rows)
    dev_shingles = set(shingles(r) for r in frozen)
    maximum = max(len(a & c)/max(1,len(a | c)) for a in train_shingles for c in dev_shingles)
    require(maximum < 0.75, 'near-duplicate train/development evidence')
    # Intentional same-evidence quartets and controlled pairs are declared; report
    # other exact duplicates rather than claiming canonical-state diversity.
    counts = Counter(tuple(e['fact'] for e in r['evidence']) for r in rows if 'handoff_group' not in r)
    return dict(pack_sha256=b.digest(pack), development_sha256=b.digest(frozen), train_cases=len(rows),
                contrast_pairs=len(pairs), handoff_groups=len(groups),
                factorial_combinations=96 if arm in ('D','E') else None,
                cross_split_max_trigram_jaccard=maximum,
                repeated_evidence_bundles=sum(v-1 for v in counts.values() if v>1),
                state_counts=dict(Counter(r['semantic_root'] for r in rows)))


def validate_config(cfg, parent):
    old = json.loads(b.common.CFG_PATH.read_text())
    require(cfg['required_parent_adapter_sha256'] == old['required_parent_adapter_sha256'] == parent['model']['adapter_sha256'], 'parent adapter drift')
    require(cfg['required_base_revision'] == parent['model']['revision'] == '51dd4bc2ade4059a6bd87649d68aa11e4fb2529b', 'base revision drift')
    require(parent['governance']['weights_frozen'] is True, 'parent must remain frozen')
    for k in ('learning_rate','weight_decay','seed','lora_rank','lora_alpha','lora_dropout','precision','maximum_optimizer_steps','gradient_accumulation_steps','max_sequence_length','checkpoint_steps'):
        require(cfg['training'][k] == old['training'][k], f'training comparison drift: {k}')
    require(cfg['development_gates'] == old['development_gates'], 'scientific acceptance gates changed')
    require(cfg['governance']['paid_training_allowed'] is False, 'new paid training requires approval')
    require(cfg['governance']['paid_checkpoint_evaluation_allowed'] is False, 'new paid evaluation requires approval')
    for k in ('parent_adapter_mutation_allowed','burned_sealed_partition_use_allowed','certification_claim_allowed','production_promotion_allowed','automatic_role_dispatch_allowed'):
        require(cfg['governance'][k] is False, f'governance drift: {k}')
    return True


def main():
    cfg = json.loads((b.common.ROOT/'configs/experiments/hephaestus_evaluator_semantic_repair_v2.json').read_text())
    parent = json.loads((b.common.ROOT/cfg['parent_candidate_manifest']).read_text())
    validate_config(cfg, parent)
    reports = {}
    for arm in 'BCDE':
        pack = b.build_pack(arm)
        reports[arm] = validate_pack(pack)
        selected, meta = b.schedule(pack, cfg)
        require(Counter(r['semantic_root'] for r in selected) == Counter({s:96 for s in b.STATE_ORDER}), 'selected state imbalance')
        reports[arm]['schedule'] = meta
        require(reports[arm]['pack_sha256'] == cfg['arms'][arm]['pack_sha256'], 'configured pack hash drift')
        require(b.digest(selected) == cfg['arms'][arm]['schedule_sha256'], 'configured schedule hash drift')
    print(json.dumps(reports, sort_keys=True))


if __name__ == '__main__':
    main()
