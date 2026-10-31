from dataclasses import replace
from types import SimpleNamespace
import pytest
from hephaestus.schemas.cpu_research import CpuResearchExecutionProfile
from hephaestus.llm.cpu_causal_research import MODEL_ID, MODEL_REVISION
from hephaestus.state.cpu_research import CpuResearchState
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.policy.internal_research import put_record
from hephaestus.control.staged_state import StagedApprovalRequest
from hephaestus.roles.cpu_research_data import InternalResearchApprovalService


def profile():
    return CpuResearchExecutionProfile('cpu', 'sha256:' + 'a'*64, MODEL_ID, MODEL_REVISION,
        {n:'sha256:'+'b'*64 for n in ('config.json','model.safetensors','tokenizer.json','tokenizer_config.json')},
        'sha256:'+'c'*64, 2, 600)


@pytest.mark.parametrize('field,value', [('device','cuda:0'),('frozen_cuda_identity',True),
    ('external_certification',True),('model_id','other'),('model_revision','main'),
    ('role_adapter','frozen-production'),('maximum_training_steps',True),
    ('maximum_training_steps',9),('maximum_cpu_seconds',float('inf')),
    ('snapshot_hashes',{}),('cpu_threads',8)])
def test_cpu_profile_cannot_downgrade_or_impersonate_frozen_cuda(field,value):
    profile().validate()
    with pytest.raises(PermissionError):
        replace(profile(), **{field:value}).validate()


def test_original_profile_and_scientific_receipts_are_immutable_on_resume(tmp_path):
    artifacts = FileSystemArtifactStore(tmp_path/'artifacts')
    repository = SQLiteStateRepository(tmp_path/'state.sqlite')
    ref = put_record(artifacts,profile().to_dict())
    state = CpuResearchState(repository,artifacts,workflow_id='wf',profile_ref=ref)
    source = put_record(artifacts,{'exact':'source'})
    state.put('operation',source,{'actual':'measurement'})
    rebuilt = CpuResearchState(repository,artifacts,workflow_id='wf',profile_ref=ref)
    assert rebuilt.get('operation',source) == {'actual':'measurement'}
    other = put_record(artifacts,{'other':'scope'})
    with pytest.raises(PermissionError):
        CpuResearchState(repository,artifacts,workflow_id='wf',profile_ref=other)
    with pytest.raises(PermissionError):
        rebuilt.get('operation',other)
    with pytest.raises(PermissionError):
        rebuilt.put('operation',source,{'forged':'measurement'})


@pytest.mark.parametrize('substep,requirements', [('action_approval_gate',('operator_approval',)),
    ('model_approval_gate',('model_approval',)),('acquisition_approval_gate',('dataset_acquisition','certify_dataset'))])
def test_internal_automatic_authority_is_never_general_approval(substep, requirements):
    ctx = SimpleNamespace(scope={'workflow_id':'wf','run_id':'run'})
    authority = InternalResearchApprovalService(ctx)
    request = StagedApprovalRequest('request','wf','run','operation','data_acquisition_audit',substep,'subject',requirements)
    assert authority.decision_for(request) is None

@pytest.mark.real_cpu
def test_actual_cpu_optimizer_resumes_from_committed_step_without_recounting(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('peft')
    from transformers import LlamaConfig, LlamaForCausalLM
    from test_generated_original_data import create_governed_source
    from hephaestus.data.generated_factory import materialize_authorized_tokens
    from hephaestus.training.cpu_research import CpuExactTokenTrainer, hash_file
    from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(['grammar'], trainers.BpeTrainer(vocab_size=300, special_tokens=['<unk>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokpath = tmp_path/'tokenizer.json'
    tok.save(str(tokpath))
    admission, candidate, review, scope = create_governed_source(tmp_path, tokpath)
    material = materialize_authorized_tokens(admission,candidate,review,**scope)
    snapshot = tmp_path/'fixture-initialized-llama-not-pretrained'
    snapshot.mkdir()
    torch.manual_seed(42)
    LlamaForCausalLM(LlamaConfig(vocab_size=tok.get_vocab_size(), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=512)).save_pretrained(snapshot)
    fixture_profile = SimpleNamespace(cpu_threads=2, seed=7, maximum_training_steps=2,
        snapshot_hashes={p.name:hash_file(p) for p in snapshot.iterdir() if p.is_file()})
    profile_ref = put_record(admission.artifacts,{'fixture_only':True})
    repository = admission.evaluation_store.repository
    state = CpuResearchState(repository,admission.artifacts,workflow_id=scope['workflow_id'],profile_ref=profile_ref)
    ctx = SimpleNamespace(state=state, root=tmp_path, materialization_ref=material, admission=admission,
        candidate=candidate,review_ref=review,scope=scope,profile=fixture_profile,profile_ref=profile_ref,
        snapshot=snapshot,repository=repository,artifacts=admission.artifacts,
        check_authority=lambda:None,check_deadline=lambda:None)
    original_append = repository.append
    def interrupt_after_commit(collection, row, **kwargs):
        saved = original_append(collection,row,**kwargs)
        if collection == 'cpu_research_optimizer_steps' and row['completed_steps'] == 1:
            raise KeyboardInterrupt('actual first step committed; simulate controller interruption')
        return saved
    monkeypatch.setattr(repository,'append',interrupt_after_commit)
    with pytest.raises(KeyboardInterrupt):
        CpuExactTokenTrainer(ctx).run('real-fixture-optimization')
    monkeypatch.setattr(repository,'append',original_append)
    observed = CpuExactTokenTrainer(ctx).run('real-fixture-optimization')
    assert observed['completed_optimizer_steps'] == 2
    assert observed['weights_changed']
    assert [r['completed_steps'] for r in repository.all('cpu_research_optimizer_steps')] == [1,2]
    assert validate_checkpoint_manifest(observed['checkpoint_ref'])[0]
    assert CpuExactTokenTrainer(ctx).run('real-fixture-optimization') == observed
    assert len(repository.all('cpu_research_optimizer_steps')) == 2
    # Rehashing a modified checkpoint cannot rewrite its immutable training receipt.
    from pathlib import Path
    from hashlib import sha256
    import json
    from hephaestus.storage.verified import canonical_bytes
    from hephaestus.training.cpu_research import verify_cpu_checkpoint
    checkpoint = Path(observed['checkpoint_ref'])
    adapter = checkpoint/'adapter_model.safetensors'
    adapter.write_bytes(adapter.read_bytes() + b'tamper')
    manifest_path = checkpoint/'checkpoint_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['components'][adapter.name] = hash_file(adapter)
    manifest['manifest_hash'] = 'sha256:' + sha256(canonical_bytes(manifest['components'])).hexdigest()
    manifest_path.write_bytes(canonical_bytes(manifest))
    assert validate_checkpoint_manifest(checkpoint)[0]
    with pytest.raises(ValueError,match='immutable optimizer receipt'):
        verify_cpu_checkpoint(observed)
