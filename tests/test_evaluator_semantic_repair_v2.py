"""Scientific invariants and failure-injection tests, all CPU-only."""
import copy
import json
from collections import Counter
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_evaluator_semantic_repair_v2 as b
import evaluator_semantic_v2_supervision as supervision
import evaluator_semantic_v2_metrics as metrics
from validate_evaluator_semantic_repair_v2 import validate_pack,validate_config
from prepare_evaluator_semantic_repair_v2 import CFG_PATH
from hephaestus.schemas.evaluator_factors import ScientificFactors,combination_space
from hephaestus.policy.evaluator_semantic_oracle import scientific_state


@pytest.fixture(scope='module')
def packs():
    return {arm:b.build_pack(arm) for arm in 'BCDE'}


@pytest.fixture(scope='module')
def cfg():
    # Negative authorization tests construct an unapproved scenario explicitly;
    # the live protocol may carry a separately recorded human approval.
    value=json.loads(CFG_PATH.read_text())
    value['governance'].update(training_approved=False,paid_training_allowed=False,paid_checkpoint_evaluation_allowed=False,budget_approval_id=None)
    return value


def test_exhaustive_oracle_truth_partition_and_frozen_semantics():
    space=combination_space()
    assert len(space)==len(set(space))==96
    # Independent disjoint truth sets, not another priority-chain implementation.
    truth={
        'scientific_rejection':{f for f in space if not f.gate_pass},
        'incomplete_evidence':{f for f in space if f.gate_pass and not f.complete},
        'inconclusive':{f for f in space if f.gate_pass and f.complete and not f.provenance_valid},
        'recheck_required':{f for f in space if f.gate_pass and f.complete and f.provenance_valid and not f.variance_stable},
    }
    eligible={f for f in space if f.gate_pass and f.complete and f.provenance_valid and f.variance_stable}
    truth.update(regressed={f for f in eligible if f.effect=='negative'},equivalent={f for f in eligible if f.effect=='neutral'},
        improved={f for f in eligible if f.effect=='positive' and not f.recheck_satisfied},
        certification_ready={f for f in eligible if f.effect=='positive' and f.recheck_satisfied})
    assert Counter({s:len(v) for s,v in truth.items()})==Counter(scientific_rejection=48,incomplete_evidence=24,inconclusive=12,recheck_required=6,regressed=2,equivalent=2,improved=1,certification_ready=1)
    for f in space:
        assert [s for s,subset in truth.items() if f in subset]==[scientific_state(f)]
        assert scientific_state(f)==b.v1._derive_state(f.to_dict())


@pytest.mark.parametrize('bad',[dict(effect='unknown'),dict(gate_pass=1),dict(complete=None)])
def test_invalid_factors_fail_closed(bad):
    values=b.v1._canonical_latent('improved');values.update(bad)
    with pytest.raises(ValueError): ScientificFactors(**values)


@pytest.mark.parametrize('arm',list('BCDE'))
def test_full_curriculum_validation_and_dose(packs,cfg,arm):
    report=validate_pack(packs[arm]);selected,_=b.schedule(packs[arm],cfg)
    assert report['pack_sha256']==cfg['arms'][arm]['pack_sha256']
    assert b.digest(selected)==cfg['arms'][arm]['schedule_sha256']
    assert len(selected)==768
    assert Counter(r['semantic_root'] for r in selected)==Counter({s:96 for s in b.STATE_ORDER})
    if arm!='B':assert report['contrast_pairs']==256
    if arm in 'DE': assert report['factorial_combinations']==96
    if arm=='E':assert report['handoff_groups']==1024


def test_dose_and_advice_are_matched_in_handoff_ablation(packs,cfg):
    treatment,_=b.schedule(packs['E'],cfg);control=b.matched_handoff_control(treatment)
    assert b.digest(control)==cfg['matched_handoff_control_sha256']
    assert len(control)==len(treatment)==768
    assert Counter(r['curriculum_family'] for r in treatment)['minimal_contrast']==512
    for a,c in zip(treatment,control):
        assert a['evidence']==c['evidence'] and a['verified_facts']==c['verified_facts'] and a['expected']==c['expected']
        if 'handoff_group' in a:assert c['upstream_output'] is None


@pytest.mark.parametrize('mutation,reason',[
    ('factor','oracle target mismatch'),('pair','contrast'),('ref','evidence-reference'),
    ('leak','identifier label leakage'),('development','immutable development')])
def test_reject_corrupt_curriculum(packs,mutation,reason):
    p=copy.deepcopy(packs['C']);r=next(x for x in p['partitions']['evaluator']['train'] if 'contrast_group' in x)
    if mutation=='factor':r['expected']['decision']='bogus'
    elif mutation=='pair':r['changed_variable']='effect' if r['changed_variable']!='effect' else 'complete'
    elif mutation=='ref':r['evidence'][0]['ref']='bad'
    elif mutation=='leak':r['context']='improved'
    else:p['partitions']['evaluator']['development_diagnostic'][0]['evidence'][0]['fact']='changed'
    with pytest.raises(ValueError,match=reason):validate_pack(p)


def test_reject_handoff_evidence_mutation(packs):
    p=copy.deepcopy(packs['E']);r=next(x for x in p['partitions']['evaluator']['train'] if 'handoff_group' in x)
    r['evidence'][0]['fact']='changed observation'
    with pytest.raises(ValueError,match='handoff evidence'):validate_pack(p)


def test_targets_equal_effective_legacy_supervision_not_alleged_generic(packs):
    import evaluator_semantic_repair_train as legacy
    assert legacy.v2.v1.target_answer.__name__=='target_answer_v3'
    rows=packs['B']['partitions']['evaluator']['train']
    for row in rows:
        assert supervision.target_answer(row)==legacy.v2.v1.target_answer(row)
        assert json.loads(supervision.target_answer(row))['rationale']==row['target_rationale']


def test_target_does_not_depend_on_import_mutations(packs,monkeypatch):
    row=packs['C']['partitions']['evaluator']['train'][0]
    expected=supervision.target_answer(row)
    monkeypatch.setattr(supervision.v1,'target_answer',lambda _:json.dumps(dict(rationale='wrong')))
    assert supervision.target_answer(row)==expected
    broken=copy.deepcopy(row);broken.pop('target_rationale')
    with pytest.raises(ValueError):supervision.target_answer(broken)


@pytest.mark.parametrize('field',['required_base_revision','required_parent_adapter_sha256','development_gates','learning_rate'])
def test_config_rejects_lineage_or_science_drift(cfg,field):
    parent=json.loads((b.common.ROOT/cfg['parent_candidate_manifest']).read_text())
    assert validate_config(cfg,parent)
    broken=copy.deepcopy(cfg)
    if field=='learning_rate':broken['training'][field]=0.001
    elif field=='development_gates':broken[field]={}
    else:broken[field]='wrong'
    with pytest.raises(ValueError):validate_config(broken,parent)


def test_paid_training_refused_before_loading(cfg):
    from run_evaluator_semantic_repair_v2 import authorize
    with pytest.raises(PermissionError,match='not approved'):authorize(cfg,{'arm':'C'})


def rows_for(pack,cases,change=None):
    result=[]
    for case in cases:
        raw=supervision.target_answer(case)
        if change:raw=change(case,raw)
        result.append(dict(case_id=case['case_id'],score=metrics.score(pack,case,raw)))
    return result


def test_perfect_metrics_include_pair_and_handoff_denominators(packs):
    p=packs['E'];allrows=p['partitions']['evaluator']['train']
    pair=next(r['contrast_group'] for r in allrows if 'contrast_group' in r)
    group=next(r['handoff_group'] for r in allrows if 'handoff_group' in r)
    cases=[r for r in allrows if r.get('contrast_group')==pair or r.get('handoff_group')==group]
    result=metrics.summarize(cases,rows_for(p,cases))
    assert result['summary']['decision_exact_pass_rate']==1
    assert result['contrast_pair_accuracy']==result['handoff_invariant_correctness']==1
    assert result['handoff_invariance']==1


def test_wrong_invariant_answers_are_not_scientific_success(packs):
    p=packs['E'];allrows=p['partitions']['evaluator']['train'];group=next(r['handoff_group'] for r in allrows if 'handoff_group' in r)
    cases=[r for r in allrows if r.get('handoff_group')==group]
    wrong=next(s for s in b.STATE_ORDER if s!=cases[0]['expected']['decision'])
    def mutate(c,raw):
        x=json.loads(raw);x['decision']=wrong;return json.dumps(x)
    result=metrics.summarize(cases,rows_for(p,cases,mutate))
    assert result['handoff_invariance']==1 and result['handoff_invariant_correctness']==0
    assert result['summary']['decision_exact_pass_rate']==0
    assert not result['frozen_development_gates']['passed']


def test_missing_duplicate_or_partial_results_fail(packs):
    p=packs['C'];cases=[r for r in p['partitions']['evaluator']['train'] if 'contrast_group' in r][:2]
    rows=rows_for(p,cases)
    with pytest.raises(ValueError):metrics.summarize(cases,rows[:1])
    with pytest.raises(ValueError):metrics.summarize(cases,rows+[rows[0]])
    with pytest.raises(ValueError,match='partial contrast'):metrics.summarize(cases[:1],rows[:1])


@pytest.mark.parametrize('mutation',[lambda x:x.update(confidence=True),lambda x:x.update(evidence_refs=[7]),lambda x:x.update(uncertainties=[None])])
def test_strict_schema_rejects_wrong_types(packs,mutation):
    p=packs['C'];case=p['partitions']['evaluator']['train'][0];x=json.loads(supervision.target_answer(case));mutation(x)
    assert not metrics.score(p,case,json.dumps(x))['schema_compliant']


def test_duplicate_json_and_unknown_refs_rejected(packs):
    p=packs['C'];case=p['partitions']['evaluator']['train'][0];raw=supervision.target_answer(case)
    bad=raw[:-1]+',"decision":"improved"}'
    assert not metrics.score(p,case,bad)['decision_exact']
    x=json.loads(raw);x['evidence_refs']=['E-NOTREAL']
    score=metrics.score(p,case,json.dumps(x))
    assert not score['evidence_grounded'] and score['hallucinated_evidence_refs']


def test_eval_cli_ignores_forged_scores_and_rejects_train_partition(packs):
    from evaluate_evaluator_semantic_v2_outputs import evaluate_outputs
    p=copy.deepcopy(packs['E']);case=p['partitions']['evaluator']['development_diagnostic'][0]
    p['partitions']['evaluator']['development_diagnostic']=[case]
    raw=json.loads(supervision.target_answer(case));raw['decision']=next(s for s in b.STATE_ORDER if s!=case['expected']['decision'])
    report=evaluate_outputs(p,'development_diagnostic',[dict(case_id=case['case_id'],raw_output=json.dumps(raw),score=dict(decision_exact=True))])
    assert report['summary']['decision_exact_pass_rate']==0
    with pytest.raises(ValueError,match='training rows'):evaluate_outputs(p,'train',[])


@pytest.mark.parametrize('prefix,full,maxlen',[
    ([1,2,3],[1,2,9,4,5],20),([1,2,3],[1,2,3],20),([1,2,3],[1,2,3,4,5],4)])
def test_loss_mask_rejects_partial_prefix_empty_target_or_truncation(packs,monkeypatch,prefix,full,maxlen):
    pytest.importorskip('torch')
    class Tok:
        def decode(self,*a,**kw):return 'unused'
    monkeypatch.setattr(supervision.v1,'template_ids',lambda tok,kind,messages,generation:prefix if generation else full)
    case=packs['C']['partitions']['evaluator']['train'][0]
    with pytest.raises(ValueError):supervision.training_example(Tok(),{'load_kind':'causal_lm'},packs['C'],case,'evaluator',maxlen)


def test_hidden_target_and_factor_metadata_do_not_enter_prompt(packs):
    case=copy.deepcopy(packs['C']['partitions']['evaluator']['train'][0])
    prompt=supervision.v1.prompt(packs['C'],case,'evaluator')
    case['expected']={k:'SENTINEL_EXPECTED' for k in case['expected']}
    case['target_rationale']='SENTINEL_RATIONALE'
    case['latent_factors']={'SENTINEL_LATENT':True}
    case['curriculum_family']='SENTINEL_FAMILY'
    assert supervision.v1.prompt(packs['C'],case,'evaluator')==prompt
    assert 'SENTINEL' not in prompt


@pytest.mark.parametrize('mode',['train','evaluate'])
def test_launcher_dry_requests_route_only_v2_without_real_secrets(cfg,mode):
    from launch_evaluator_semantic_repair_v2 import request
    body=request(cfg,'a'*40,'v2-test',mode,real_env=False)
    shell=body['dockerStartCmd'][-1]
    assert f'run_evaluator_semantic_repair_v2.py --mode {mode}' in shell
    assert 'run_evaluator_semantic_invariance_repair_v1.py' not in shell
    assert body['env']['RUNPOD_S3_SECRET_ACCESS_KEY']=='<secret>'
    assert body['env']['HEPHAESTUS_EVALUATOR_V2_RUN_ID']=='v2-test'


@pytest.mark.parametrize('price',[None,0,-1,float('nan'),2.50])
def test_missing_invalid_or_excessive_price_fails_closed(cfg,price):
    from launch_evaluator_semantic_repair_v2 import bounded_cost
    with pytest.raises(ValueError):bounded_cost({'costPerHr':price},1,cfg,'train')


def test_per_run_cost_ceiling_reserves_teardown(cfg):
    from launch_evaluator_semantic_repair_v2 import bounded_cost
    assert bounded_cost({'costPerHr':2.1},10,cfg,'train')['estimated_cost_usd']>0
    with pytest.raises(TimeoutError):bounded_cost({'costPerHr':2.1},2740,cfg,'train')
    with pytest.raises(TimeoutError):bounded_cost({'costPerHr':2.1},1540,cfg,'evaluate')


def test_teardown_requires_actual_not_found_not_any_exception(monkeypatch):
    import launch_evaluator_semantic_repair_v2 as launch
    from hephaestus.providers.runpod.execution import RunPodApiError
    class Pod:
        def __init__(self,error):self.error=error;self.deletes=0
        def delete_pod(self,pod):self.deletes+=1
        def get_pod(self,pod):raise self.error
    monkeypatch.setattr(launch.time,'sleep',lambda _:None)
    outage=Pod(RunPodApiError('transport layer failure'))
    assert not launch.delete_verified(outage,'pod')['verified_absent']
    assert outage.deletes==5
    missing=Pod(RunPodApiError('unexpected get-Pod response status: 404'))
    assert launch.delete_verified(missing,'pod')['verified_absent']


@pytest.mark.parametrize('failure',['missing_price','timeout','terminal_mismatch','success'])
def test_launcher_always_deletes_allocated_pod(cfg,monkeypatch,failure):
    import launch_evaluator_semantic_repair_v2 as launch
    approved=copy.deepcopy(cfg)
    approved['governance'].update(training_approved=True,paid_training_allowed=True,budget_approval_id='test-approval')
    manifest=dict(arm='C',scientific_config_sha256=b.digest(approved))
    monkeypatch.setenv('HEPHAESTUS_EVALUATOR_V2_AUTHORIZED','LAUNCH_APPROVED_EVALUATOR_SEMANTIC_V2')
    monkeypatch.setenv('RUNPOD_NETWORK_VOLUME_ID','test-bucket')
    monkeypatch.setattr(launch,'request',lambda *a,**kw:{'safe':'mock request'})
    reads=[];claims=[];deleted=[]
    def read(client,key):
        reads.append(key)
        if len(reads)==1:return None
        return dict(status='training_complete',repo_sha=('b' if failure=='terminal_mismatch' else 'a')*40,run_id='test')
    monkeypatch.setattr(launch.infra,'maybe_json',read)
    monkeypatch.setattr(launch,'delete_verified',lambda execution,pod:(deleted.append(pod) or dict(verified_absent=True)))
    class Client:
        def put_object(self,**kw):claims.append(kw)
    class Pod:
        def _create_pod(self,body):return {'id':'mock-pod'}
        def get_pod(self,pod):return dict(costPerHr=None if failure=='missing_price' else 2.1,desiredStatus='RUNNING')
    if failure=='timeout':
        ticks=iter([0,4000,4001]);monkeypatch.setattr(launch.time,'monotonic',lambda:next(ticks))
    if failure=='success':
        assert launch.execute(approved,manifest,'train',execution=Pod(),client=Client(),repo_sha='a'*40,run_id='test')['status']=='training_complete'
    else:
        with pytest.raises((ValueError,TimeoutError)):
            launch.execute(approved,manifest,'train',execution=Pod(),client=Client(),repo_sha='a'*40,run_id='test')
    assert deleted==['mock-pod']
    assert claims[0]['IfNoneMatch']=='*'


def test_checked_in_flags_refuse_paid_launcher_before_any_io(cfg):
    from launch_evaluator_semantic_repair_v2 import execute
    with pytest.raises(PermissionError):execute(cfg,{'arm':'C'},'train',execution=None,client=None,repo_sha='a'*40,run_id='test')


def test_conditional_claim_failure_prevents_allocation(cfg,monkeypatch):
    import launch_evaluator_semantic_repair_v2 as launch
    approved=copy.deepcopy(cfg);approved['governance'].update(training_approved=True,paid_training_allowed=True,budget_approval_id='test-approval')
    manifest=dict(arm='C',scientific_config_sha256=b.digest(approved))
    monkeypatch.setenv('HEPHAESTUS_EVALUATOR_V2_AUTHORIZED','LAUNCH_APPROVED_EVALUATOR_SEMANTIC_V2')
    monkeypatch.setenv('RUNPOD_NETWORK_VOLUME_ID','test-bucket')
    monkeypatch.setattr(launch,'request',lambda *a,**kw:{})
    monkeypatch.setattr(launch.infra,'maybe_json',lambda *a:None)
    class Claim:
        def put_object(self,**kw):raise RuntimeError('conditional claim rejected')
    class NoPod:
        def _create_pod(self,*a):pytest.fail('claim failure must prevent allocation')
    with pytest.raises(RuntimeError,match='conditional claim rejected'):
        launch.execute(approved,manifest,'train',execution=NoPod(),client=Claim(),repo_sha='a'*40,run_id='test')


def test_checkpoint_evaluation_is_separately_gated(cfg):
    from run_evaluator_semantic_repair_v2 import evaluate
    with pytest.raises(PermissionError,match='evaluation is not approved'):
        evaluate(cfg,None,None,None,None,client=None,run_id='test',repo_sha='a'*40,root=None)


@pytest.mark.parametrize('arm',list('BCDE'))
def test_each_checkpoint_block_preserves_state_balance(packs,cfg,arm):
    selected,meta=b.schedule(packs[arm],cfg)
    for index in range(6):
        block=selected[index*128:(index+1)*128]
        assert Counter(r['semantic_root'] for r in block)==Counter({s:16 for s in b.STATE_ORDER})
        if 'checkpoint_blocks' in meta:
            declared=meta['checkpoint_blocks'][index]
            assert declared['state_counts']==dict(Counter(r['semantic_root'] for r in block))
            if 'family_counts' in declared:
                assert declared['family_counts']==dict(Counter(r['curriculum_family'] for r in block))


def test_paid_flags_alone_do_not_authorize_a_new_budget(cfg,monkeypatch):
    from run_evaluator_semantic_repair_v2 import authorize
    approved=copy.deepcopy(cfg);approved['governance'].update(training_approved=True,paid_training_allowed=True)
    monkeypatch.setenv('HEPHAESTUS_EVALUATOR_V2_AUTHORIZED','LAUNCH_APPROVED_EVALUATOR_SEMANTIC_V2')
    with pytest.raises(PermissionError,match='budget approval identity'):
        authorize(approved,dict(arm='C',scientific_config_sha256=b.digest(approved)))


def test_new_workflow_is_explicit_marker_only_and_default_protocol_disabled(cfg):
    workflow=(b.common.ROOT/'.github/workflows/evaluator-semantic-v2-approved-proof.yml').read_text()
    assert "if: contains(github.event.head_commit.message, '[launch-approved-evaluator-semantic-v2-C]')" in workflow
    assert 'workflow_dispatch' not in workflow and 'matrix:' not in workflow
    assert not any(cfg['governance'][k] for k in ('training_approved','paid_training_allowed','paid_checkpoint_evaluation_allowed'))
    assert cfg['governance']['budget_approval_id'] is None


def test_live_activation_matches_recorded_human_budget():
    live=json.loads(CFG_PATH.read_text())
    gov=live['governance']
    if any(gov[k] for k in ('training_approved','paid_training_allowed','paid_checkpoint_evaluation_allowed')):
        approval=json.loads((b.common.ROOT/gov.get('approval_record_path','configs/experiments/evaluator_semantic_repair_v2.approval.json')).read_text())
        assert approval['status']=='approved' and gov['budget_approval_id']==approval['approval_id']
        assert approval['maximum_total_usd']==2.50
        assert live['execution']['max_estimated_total_usd_train']==approval['maximum_training_usd']
        assert live['execution']['max_estimated_total_usd_evaluate']==approval['maximum_evaluation_usd']
        assert live['execution']['max_hourly_usd_per_pod']==approval['maximum_hourly_usd']
        assert not approval['additional_ablation_runs_authorized']
        assert not approval['production_or_certification_authorized']


@pytest.mark.parametrize('rate',[2.1,2.49,0,None])
def test_catalog_preflight_prevents_overprice_before_allocation(cfg,monkeypatch,rate):
    from launch_evaluator_semantic_repair_v2 import quote_gpu
    import io,urllib.request
    monkeypatch.setenv('RUNPOD_API_KEY','test-only-token')
    data={'data':{'gpuTypes':[{'id':cfg['execution']['gpu_type_ids'][0],'lowestPrice':{'uninterruptablePrice':rate,'stockStatus':'High'}}]}}
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(json.dumps(data).encode()))
    if rate is not None and 0<rate<=cfg['execution']['max_hourly_usd_per_pod']:assert quote_gpu(cfg)['catalog_floor_usd_per_hour']==rate
    else:
        with pytest.raises(ValueError):quote_gpu(cfg)
