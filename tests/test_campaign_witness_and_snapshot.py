"""Real process deaths, retained commits, complete backups and stale denial."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from hephaestus.schemas.storage_admission import LocalStorageProfile
from hephaestus.state.campaign_snapshot import snapshot_campaign, restore_campaign, verify_snapshot
from hephaestus.state.campaign_witness import CampaignRollbackError
from hephaestus.state.execution_ownership import execution_owner
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository, StateIntegrityError, StateCommitUncertain


def campaign(root):
    root.mkdir(parents=True, exist_ok=True)
    return SQLiteStateRepository(root/'state'/'production.sqlite3', campaign_witness=True)


def run_child(code, *args):
    result = subprocess.run([sys.executable, '-c', code, *map(str, args)],
                            capture_output=True, text=True, timeout=30)
    return result


@pytest.mark.parametrize('phase,expected_count', [
    ('before', 0), ('body', 0), ('prepared', 0), ('acknowledgement', 1)])
def test_killed_real_application_transaction_has_one_classified_outcome(tmp_path, phase, expected_count):
    root = tmp_path/'campaign'
    repository = campaign(root)
    code = '''
import os,sys
from pathlib import Path
from hephaestus.storage.sqlite import SQLiteStateRepository
r=SQLiteStateRepository(Path(sys.argv[1]),initialize=False)
w=r._db.witness()
phase=sys.argv[2]
if phase=='before':os._exit(71)
if phase=='prepared':
    original=w.prepare
    def die(*args):
        original(*args)
        os._exit(71)
    w.prepare=die
if phase=='acknowledgement':w.committed=lambda *args:os._exit(71)
with r.transaction() as tx:
    tx.append('scientific_publications',{'result':'durable'},operation_id='scientific-action')
    if phase=='body':os._exit(71)
'''
    killed = run_child(code, repository.path, phase)
    assert killed.returncode == 71, killed.stderr
    reopened = SQLiteStateRepository(repository.path, initialize=False)
    assert len(reopened.all('scientific_publications')) == expected_count
    # Starting a supported operation reconciles from the ORIGINAL live store.
    with execution_owner(reopened, 'research', 'restart'):
        reopened.append('scientific_publications', {'result':'durable'}, operation_id='scientific-action')
    assert len(reopened.all('scientific_publications')) == 1
    assert reopened._db.witness().verify_chain()['record']['kind'] == 'committed'


def test_crash_during_reconciliation_remains_consumed_and_recovers_once(tmp_path):
    repository = campaign(tmp_path/'campaign')
    code = '''
import os,sys
from pathlib import Path
from hephaestus.storage.sqlite import SQLiteStateRepository
r=SQLiteStateRepository(Path(sys.argv[1]),initialize=False)
r._db.witness().committed=lambda *args:os._exit(71)
r.append('exposures',{'status':'consumed'},operation_id='one-use')
'''
    assert run_child(code, repository.path).returncode == 71
    recovery = '''
import os,sys
from pathlib import Path
from hephaestus.storage.sqlite import SQLiteStateRepository
r=SQLiteStateRepository(Path(sys.argv[1]),initialize=False)
w=r._db.witness()
w._append=lambda *args:os._exit(72)
r.append('exposures',{'status':'consumed'},operation_id='one-use')
'''
    assert run_child(recovery, repository.path).returncode == 72
    resumed = SQLiteStateRepository(repository.path, initialize=False)
    resumed.append('exposures', {'status':'consumed'}, operation_id='one-use')
    assert len(resumed.all('exposures')) == 1


def test_unknown_witness_ack_retains_operation_identity_and_resolves_without_duplicate(tmp_path, monkeypatch):
    repository = campaign(tmp_path/'campaign')
    witness = repository._db.witness()
    with monkeypatch.context() as patch:
        patch.setattr(witness, 'committed', lambda *args: (_ for _ in ()).throw(OSError('ack lost')))
        with pytest.raises(StateCommitUncertain) as failure:
            repository.append('science', {'value':42}, operation_id='result-42')
        assert failure.value.operation_ids == ('result-42',)
    repository.append('science', {'value':42}, operation_id='result-42')
    assert len(repository.all('science')) == 1


def test_missing_or_torn_witness_cannot_silently_enroll_existing_science(tmp_path):
    repository = campaign(tmp_path/'campaign')
    repository.append('science', {'v':1})
    witness = repository._db.witness()
    original = witness.journal.read_bytes()
    witness.journal.rename(witness.root/'retained-original')
    with pytest.raises((StateIntegrityError, CampaignRollbackError), match='witness'):
        SQLiteStateRepository(repository.path)
    witness.journal.write_bytes(original+b'{"torn"')
    with pytest.raises(CampaignRollbackError, match='torn'):
        SQLiteStateRepository(repository.path)
    assert (witness.root/'retained-original').read_bytes() == original


def test_nonempty_unwitnessed_legacy_state_requires_explicit_evidence_migration(tmp_path):
    path = tmp_path/'legacy.sqlite3'
    legacy = SQLiteStateRepository(path)
    legacy.append('science', {'historical':'never silently adopted'})
    with pytest.raises(CampaignRollbackError, match='automatically enrolled'):
        SQLiteStateRepository(path, campaign_witness=True)
    assert legacy.all('science') == [{'historical':'never silently adopted'}]


def test_whole_campaign_backup_includes_live_wal_artifacts_and_new_process_restore(tmp_path):
    root = tmp_path/'campaign'
    repository = campaign(root)
    artifact = FileSystemArtifactStore(root/'artifacts').put_bytes(b'actual retained evidence')
    reader = sqlite3.connect(repository.path)
    try:
        reader.execute('BEGIN')
        reader.execute('SELECT count(*) FROM state_records').fetchone()
        for index in range(25):
            repository.append('science', {'index':index, 'artifact_ref':artifact.artifact_ref}, operation_id=f'op-{index}')
        assert Path(str(repository.path)+'-wal').stat().st_size > 0
        saved = snapshot_campaign(root, repository, tmp_path/'backup')
    finally:
        reader.close()
    manifest = verify_snapshot(tmp_path/'backup', expected_manifest_sha256=saved['manifest_sha256'])
    assert manifest['relationships']['state_records'] == 25
    assert not manifest['relocatable'] and not manifest['independent_exposure_authority']
    root.rename(tmp_path/'quarantined')
    restored = restore_campaign(tmp_path/'backup', root, expected_manifest_sha256=saved['manifest_sha256'])
    cold = run_child('''
import json,sys
from pathlib import Path
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.state.execution_ownership import execution_owner
r=SQLiteStateRepository(Path(sys.argv[1]),initialize=False)
with execution_owner(r,'cold','resume'):
    assert len(r.all('science'))==25
    r.append('science',{'resumed':True},operation_id='resumed')
print(json.dumps({'rows':len(r.all('science')),'identity':r.storage_identity()}))
''', root/'state'/'production.sqlite3')
    assert cold.returncode == 0, cold.stderr
    assert json.loads(cold.stdout)['rows'] == 26
    assert restored['storage_identity'] == manifest['storage_identity']
    with pytest.raises(StateIntegrityError, match='file identity changed'):
        repository.append('science', {'stale_handle':True})


def test_old_backup_cannot_restore_consumed_permissions_or_create_empty_replacement(tmp_path):
    root = tmp_path/'campaign'
    repository = campaign(root)
    saved = snapshot_campaign(root, repository, tmp_path/'backup')
    repository.append('exposures', {'status':'consumed'}, operation_id='single-assessment')
    root.rename(tmp_path/'retained-live-store')
    with pytest.raises(CampaignRollbackError, match='rollback'):
        restore_campaign(tmp_path/'backup', root, expected_manifest_sha256=saved['manifest_sha256'])
    with pytest.raises(CampaignRollbackError):
        campaign(root)
    assert not (root/'state'/'production.sqlite3').exists()
    retained = SQLiteStateRepository(tmp_path/'retained-live-store'/'state'/'production.sqlite3', read_only=True)
    assert retained.all('exposures') == [{'status':'consumed'}]


def test_unresolved_intent_cannot_be_resolved_by_presenting_old_backup(tmp_path):
    root = tmp_path/'campaign'
    repository = campaign(root)
    saved = snapshot_campaign(root, repository, tmp_path/'backup')
    code = '''
import os,sys
from pathlib import Path
from hephaestus.storage.sqlite import SQLiteStateRepository
r=SQLiteStateRepository(Path(sys.argv[1]),initialize=False)
r._db.witness().committed=lambda *args:os._exit(71)
r.append('exposures',{'status':'consumed'})
'''
    assert run_child(code, repository.path).returncode == 71
    root.rename(tmp_path/'original-for-reconciliation')
    with pytest.raises(CampaignRollbackError, match='unresolved'):
        restore_campaign(tmp_path/'backup', root, expected_manifest_sha256=saved['manifest_sha256'])


def test_snapshot_denies_live_worker_missing_evidence_and_modified_bundle(tmp_path):
    root = tmp_path/'campaign'
    repository = campaign(root)
    worker = root/'training'/'job'/'handle.json'
    worker.parent.mkdir(parents=True)
    worker.write_text(json.dumps({'status':'running'}))
    with pytest.raises(StateIntegrityError, match='settled'):
        snapshot_campaign(root, repository, tmp_path/'backup')
    worker.unlink()
    artifact = FileSystemArtifactStore(root/'artifacts').put_bytes(b'evidence')
    repository.append('science', {'ref':artifact.artifact_ref})
    saved = snapshot_campaign(root, repository, tmp_path/'backup')
    stored = tmp_path/'backup'/'payload'/'artifacts'/'objects'/'sha256'/artifact.content_hash[:2]/artifact.content_hash
    stored.write_bytes(b'altered')
    with pytest.raises(StateIntegrityError, match='file set'):
        verify_snapshot(tmp_path/'backup', expected_manifest_sha256=saved['manifest_sha256'])
    Path(artifact.storage_path).unlink()
    with pytest.raises(StateIntegrityError, match='absent'):
        snapshot_campaign(root, repository, tmp_path/'backup2')


def test_external_fence_cannot_be_replaced_by_renaming_campaign_root(tmp_path):
    root = tmp_path/'campaign'
    repository = campaign(root)
    with execution_owner(repository, 'controller', 'first'):
        result = run_child('''
import sys
from pathlib import Path
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.state.execution_ownership import execution_owner
from hephaestus.state.single_writer import WriterBusyError
r=SQLiteStateRepository(Path(sys.argv[1]),initialize=False)
try:
    with execution_owner(r,'controller','second'):
        r.append('science',{'duplicate':True})
except WriterBusyError:sys.exit(73)
''', repository.path)
        assert result.returncode == 73, result.stderr
    assert repository.all('science') == []


@pytest.mark.parametrize('phase', ['backup_copy', 'backup_seal', 'restore_copy', 'restore_publish'])
def test_process_death_during_backup_or_restore_never_publishes_incomplete_campaign(tmp_path, phase):
    root = tmp_path/'campaign'
    repository = campaign(root)
    repository.append('science', {'value':'retained'}, operation_id='one')
    (root/'input.json').write_text('{"public":true}')
    backup = tmp_path/'backup'
    saved = snapshot_campaign(root, repository, backup)
    code = '''
import os,sys
from pathlib import Path
import hephaestus.state.campaign_snapshot as module
from hephaestus.storage.sqlite import SQLiteStateRepository
root,backup=Path(sys.argv[1]),Path(sys.argv[2])
phase,commitment=sys.argv[3],sys.argv[4]
r=SQLiteStateRepository(root/'state'/'production.sqlite3',initialize=False)
if phase=='backup_copy':module.shutil.copyfile=lambda *a,**k:os._exit(74)
if phase=='backup_seal':module.sync_tree=lambda *a:os._exit(74)
if phase.startswith('backup'):
    module.snapshot_campaign(root,r,backup.with_name('new-backup'))
else:
    if phase=='restore_copy':module.shutil.copytree=lambda *a,**k:os._exit(74)
    if phase=='restore_publish':module.sync_tree=lambda *a:os._exit(74)
    module.restore_campaign(backup,root,expected_manifest_sha256=commitment,
        quarantine_existing_to=root.with_name('quarantine'))
'''
    killed = run_child(code, root, backup, phase, saved['manifest_sha256'])
    assert killed.returncode == 74, killed.stderr
    if phase.startswith('backup'):
        assert not (tmp_path/'new-backup').exists()
        assert repository.all('science') == [{'value':'retained'}]
    else:
        assert not root.exists()
        assert (tmp_path/'quarantine'/'state'/'production.sqlite3').exists()
        with pytest.raises(CampaignRollbackError):
            campaign(root)
        # The sealed latest snapshot can recover; partial install is never
        # consulted as state or used to infer any missing completion.
        root.rmdir()
        restore_campaign(backup, root, expected_manifest_sha256=saved['manifest_sha256'])
        cold = SQLiteStateRepository(root/'state'/'production.sqlite3', initialize=False)
        with execution_owner(cold, 'restart', 'verified'):
            assert cold.all('science') == [{'value':'retained'}]


def test_restore_checks_checkpoint_components_and_preserves_same_absolute_references(tmp_path):
    from hephaestus.schemas.checkpoint_identity import CheckpointIdentity
    from hephaestus.state.checkpoint_identity_store import CheckpointIdentityStore
    from hephaestus.storage.verified import canonical_bytes
    root = tmp_path/'campaign'
    repository = campaign(root)
    checkpoint = root/'artifacts'/'training-runs'/'run'/'checkpoint-1'
    checkpoint.mkdir(parents=True)
    components = {'model.bin':'sha256:'+hashlib.sha256(b'model').hexdigest(),
                  'optimizer.pt':'sha256:'+hashlib.sha256(b'optimizer').hexdigest()}
    (checkpoint/'model.bin').write_bytes(b'model')
    (checkpoint/'optimizer.pt').write_bytes(b'optimizer')
    identity = 'sha256:'+hashlib.sha256(canonical_bytes(components)).hexdigest()
    (checkpoint/'checkpoint_manifest.json').write_bytes(canonical_bytes({
        'components':components,'manifest_hash':identity,'partial_write':False}))
    CheckpointIdentityStore(repository).retain(CheckpointIdentity('lineage',str(checkpoint),identity,'producer'))
    saved = snapshot_campaign(root, repository, tmp_path/'backup')
    restore_campaign(tmp_path/'backup', root, expected_manifest_sha256=saved['manifest_sha256'],
                     quarantine_existing_to=tmp_path/'quarantine')
    reopened = SQLiteStateRepository(root/'state'/'production.sqlite3', initialize=False)
    assert CheckpointIdentityStore(reopened).get('lineage',str(checkpoint)).manifest_hash == identity
    (checkpoint/'optimizer.pt').write_bytes(b'bad')
    with pytest.raises(StateIntegrityError, match='checkpoint'):
        snapshot_campaign(root, reopened, tmp_path/'bad-backup')


def test_corrupt_state_blocks_operation_before_any_research_or_witness_reset(tmp_path):
    root = tmp_path/'campaign'
    repository = campaign(root)
    repository.append('science', {'value':'retained'})
    original_witness = repository._db.witness().journal.read_bytes()
    original_bytes = repository.path.read_bytes()  # no concurrent SQL use
    repository.path.write_bytes(b'corrupt header'+original_bytes[14:])
    with pytest.raises(StateIntegrityError):
        with execution_owner(repository, 'controller', 'resume'):
            pytest.fail('corrupt state admitted research')
    assert repository._db.witness().journal.read_bytes() == original_witness


def test_json_lineage_mutation_uses_the_same_outside_campaign_fence(tmp_path):
    from hephaestus.state.campaign_witness import bind_state_directory
    root = tmp_path/'campaign'
    repository = campaign(root)
    bind_state_directory(root/'state', repository)
    with execution_owner(repository, 'controller', 'held'):
        # A JSON mutation must wait for the campaign owner, rather than race a
        # snapshot. Kill the blocked process; no replacement lineage is written.
        child = subprocess.Popen([sys.executable,'-c','''
import sys
from pathlib import Path
from hephaestus.state.lineage_store import LineageStore
LineageStore(Path(sys.argv[1])).set_current({'lineage_id':'new','stage_name':'smoke_test'})
''',str(root/'state')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                child.wait(timeout=.3)
            assert not (root/'state'/'lineage_states.json').exists()
        finally:
            child.kill(); child.communicate(timeout=10)


def test_read_only_inspector_cannot_be_used_to_execute_research(tmp_path):
    repository = campaign(tmp_path/'campaign')
    inspector = SQLiteStateRepository(repository.path, read_only=True)
    assert inspector.all('science') == []
    with pytest.raises(PermissionError, match='inspectors'):
        with execution_owner(inspector, 'research', 'forbidden'):
            pytest.fail('read-only inspection was used as execution custody')


def test_quiescence_uses_acknowledged_worker_exit_not_reused_or_foreign_namespace_pid(tmp_path):
    root=tmp_path/'campaign';repository=campaign(root)
    handle=root/'training'/'job'/'handle.json';handle.parent.mkdir(parents=True)
    # The old numeric PID now names this inspector, not the reaped worker.
    handle.write_text(json.dumps({'status':'completed','metadata':{'pid':os.getpid(),'returncode':0}}))
    snapshot_campaign(root,repository,tmp_path/'verified')
    # A disappeared PID in another namespace does not prove that an unknown
    # worker exited; an observed service-level exit acknowledgement is required.
    handle.write_text(json.dumps({'status':'failed','metadata':{'pid':99999999}}))
    with pytest.raises(StateIntegrityError,match='unacknowledged'):
        snapshot_campaign(root,repository,tmp_path/'unknown-worker')
