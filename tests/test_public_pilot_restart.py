"""Real retained state and lost-ack reconciliation; generation is a software fixture."""
from pathlib import Path
import hashlib
import pytest
from hephaestus.schemas.public_pilot import PublicPilotMeasurement
from hephaestus.state.public_pilot_store import PublicPilotGenerationStore
from hephaestus.storage.sqlite import SQLiteStateRepository,StateCommitUncertain
from hephaestus.storage.filesystem import FileSystemArtifactStore


def request():
    index=0
    while True:
        identity='sha256:'+hashlib.sha256(f'public-software-fixture-{index}'.encode()).hexdigest()
        if int(identity[7:],16)%10==8:break
        index+=1
    return PublicPilotMeasurement('baseline','sha256:'+'1'*64,identity,'sha256:'+'2'*64,
        'sha256:'+'3'*64,'sha256:'+'4'*64,'development',128)


def test_restart_returns_identical_committed_public_sample_without_generation(tmp_path):
    path=tmp_path/'campaign'/'state.sqlite3'
    artifacts=FileSystemArtifactStore(tmp_path/'campaign'/'artifacts')
    req=request();calls=[]
    def produce():calls.append(1);return {'sample_id':req.sample_id,'generated_tokens':[3,2],'eos':True}
    first=PublicPilotGenerationStore(SQLiteStateRepository(path,campaign_witness=True),artifacts).measure(req,produce)
    reopened=PublicPilotGenerationStore(SQLiteStateRepository(path,initialize=False),artifacts)
    assert reopened.measure(req,produce)==first and calls==[1]
    assert len(reopened.repository.all('public_pilot_measurements'))==1


def test_unknown_ack_reconciles_the_committed_raw_prefix_without_reexecuting(tmp_path,monkeypatch):
    from hephaestus.storage import sqlite
    repository=SQLiteStateRepository(tmp_path/'state.sqlite3',campaign_witness=True)
    store=PublicPilotGenerationStore(repository,FileSystemArtifactStore(tmp_path/'artifacts'))
    req=request();calls=[]
    def produce():calls.append(1);return {'sample_id':req.sample_id,'generated_tokens':[3,2],'eos':True}
    with monkeypatch.context() as patch:
        patch.setattr(sqlite,'sync_directory_chain',lambda *args:(_ for _ in ()).throw(OSError('lost SQL ack')))
        with pytest.raises(StateCommitUncertain):store.measure(req,produce)
    assert store.measure(req,produce)['eos'] and calls==[1]


def test_scope_rejects_final_partition_and_confirmatory_purpose_before_compute():
    req=request();req.purpose='confirmatory'
    with pytest.raises(PermissionError):req.__post_init__()
    req.purpose='nonconfirmatory-public-development';req.sample_id='sha256:'+format(9,'064x')
    with pytest.raises(PermissionError):req.__post_init__()


def test_resume_cannot_reconstruct_missing_training_or_launch_an_optimizer(tmp_path,monkeypatch):
    pytest.importorskip('torch');pytest.importorskip('transformers')
    from hephaestus.development.public_smol_pilot import resume_public_pilot,_publish
    from hephaestus.training.hf_lifecycle import TransformersTrainingLifecycleService
    root=tmp_path/'campaign'
    repository=SQLiteStateRepository(root/'state'/'production.sqlite3',campaign_witness=True)
    artifacts=FileSystemArtifactStore(root/'artifacts')
    ref=_publish(repository,artifacts,'contract',{'public':'incomplete software fixture'})
    monkeypatch.setattr(TransformersTrainingLifecycleService,'launch',lambda *args:pytest.fail('optimizer launched while resuming'))
    with pytest.raises(PermissionError,match='completed training'):
        resume_public_pilot(root,expected_contract_ref=ref)


def test_missing_witness_blocks_an_existing_low_level_controller_before_compute(tmp_path):
    from hephaestus.state.execution_ownership import execution_owner
    from hephaestus.storage.sqlite import StateIntegrityError
    repository=SQLiteStateRepository(tmp_path/'state.sqlite3',campaign_witness=True)
    journal=repository._db.witness().journal;retained=journal.with_name('retained-for-test.jsonl')
    journal.rename(retained)
    try:
        restarted=SQLiteStateRepository(repository.path,initialize=False)
        with pytest.raises(StateIntegrityError,match='witness'):
            with execution_owner(restarted,'research','must-not-start'):
                pytest.fail('research started with a missing retained witness')
    finally:
        retained.rename(journal)


def test_phase_checkpoint_seals_raw_state_and_retains_its_current_control_journal(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('transformers')
    import json
    from hephaestus.development.public_smol_pilot import phase_snapshot_writer
    from hephaestus.state.campaign_snapshot import verify_snapshot
    from hephaestus.storage.verified import canonical_bytes
    root=tmp_path/'campaign';repository=SQLiteStateRepository(root/'state'/'production.sqlite3',campaign_witness=True)
    (root/'pilot-contract.json').write_bytes(canonical_bytes({'purpose':'public software checkpoint fixture'}))
    phase_snapshot_writer(tmp_path/'outside-checkpoints')('fixture-ready',repository)
    phase=tmp_path/'outside-checkpoints'/'001-fixture-ready'
    receipt=json.loads((phase/'checkpoint.json').read_text())
    assert hashlib.sha256((phase/'separately-retained-control-journal.jsonl').read_bytes()).hexdigest()==receipt['control_journal_sha256'][7:]
    assert verify_snapshot(phase/'complete-snapshot',expected_manifest_sha256=receipt['manifest_sha256'])['storage_identity']==repository.storage_identity()
