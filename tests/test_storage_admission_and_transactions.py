from contextlib import closing
import os
from pathlib import Path
import sqlite3

import pytest

from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.storage.sqlite import SQLiteStateRepository, StateIntegrityError


@pytest.mark.parametrize('composed',[False,True])
def test_unadmitted_path_is_rejected_before_any_directory_creation(monkeypatch,composed):
    root=Path('/unverified-research-volume/campaign')
    original=Path.mkdir
    def no_mutation(path,*args,**kwargs):
        if path==root or root in path.parents:
            raise AssertionError('unadmitted storage was mutated before admission')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'mkdir',no_mutation)
    with pytest.raises(PermissionError,match='storage'):
        if composed:
            ProductionCompositionRoot(ProductionCompositionSettings(root/'state',root/'artifacts')).build()
        else: SQLiteStateRepository(root/'state.sqlite3')


def test_symlinked_database_alias_is_not_an_admitted_write_path(tmp_path):
    actual=tmp_path/'actual';actual.mkdir()
    alias=tmp_path/'alias';alias.symlink_to(actual,target_is_directory=True)
    with pytest.raises(PermissionError,match='symlink'):
        SQLiteStateRepository(alias/'state.sqlite3')
    assert not (actual/'state.sqlite3').exists()


def test_replacing_a_healthy_database_does_not_rebind_a_live_repository(tmp_path):
    path=tmp_path/'state.sqlite3';repository=SQLiteStateRepository(path)
    repository.append('publications',{'id':'one'},operation_id='published-one')
    backup=tmp_path/'replacement.sqlite3';repository.snapshot_to(backup)
    os.replace(backup,path)
    with pytest.raises(StateIntegrityError,match='identity|replaced'):
        repository.append('publications',{'id':'two'},operation_id='not-authorized-for-rebound-store')
    cold=SQLiteStateRepository(path,read_only=True)
    assert cold.all('publications')==[{'id':'one'}]


def test_schema_initialization_failure_is_one_real_transaction(tmp_path,monkeypatch):
    path=tmp_path/'incomplete.sqlite3';original=sqlite3.connect
    class DenyMigration(sqlite3.Connection):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.set_authorizer(lambda action,table,*_:(sqlite3.SQLITE_DENY if
                action==sqlite3.SQLITE_INSERT and table=='infrastructure_schema_migrations' else sqlite3.SQLITE_OK))
    def connection(*args,**kwargs):
        return original(*args,**kwargs,factory=DenyMigration)
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3,'connect',connection)
        with pytest.raises(sqlite3.DatabaseError): SQLiteStateRepository(path)
    with closing(original(path)) as c:
        names=[r[0] for r in c.execute("select name from sqlite_master where name not like 'sqlite_%'")]
    assert names==[],names


def test_commit_acknowledgement_loss_reports_the_retained_operation_identity(tmp_path,monkeypatch):
    path=tmp_path/'uncertain.sqlite3';repository=SQLiteStateRepository(path)
    original=repository._db.connect
    class LostAcknowledgement:
        def __init__(self,connection):self.connection=connection
        def __getattr__(self,name):return getattr(self.connection,name)
        def commit(self):
            self.connection.commit()
            raise sqlite3.OperationalError('injected loss after actual commit')
    with monkeypatch.context() as patch:
        patch.setattr(repository._db,'connect',lambda *a,**kw:LostAcknowledgement(original(*a,**kw)))
        with pytest.raises(RuntimeError,match='unknown') as error:
            repository.append('publications',{'id':'committed-without-ack'})
    operation_ids=error.value.operation_ids
    assert len(operation_ids)==1
    restarted=SQLiteStateRepository(path)
    rows=restarted.records('publications')
    assert len(rows)==1 and rows[0]['operation_id']==operation_ids[0]
    restarted.append('publications',{'id':'committed-without-ack'},operation_id=operation_ids[0])
    assert len(restarted.all('publications'))==1


@pytest.mark.parametrize('filesystem',['nfs','cifs','fuse.sshfs','9p','virtiofs','tmpfs'])
def test_network_unverified_and_memory_mounts_fail_closed_before_mutation(tmp_path,monkeypatch,filesystem):
    import hephaestus.policy.storage_admission as policy
    monkeypatch.setattr(policy,'_mounts',lambda:[('/',filesystem,'0:1')])
    path=tmp_path/'not-created'/'state.sqlite3'
    with pytest.raises(PermissionError,match='filesystem'):
        SQLiteStateRepository(path)
    assert not path.parent.exists()


def test_overlay_scope_is_explicitly_ephemeral_and_cannot_admit_persistent_research(tmp_path,monkeypatch):
    import hephaestus.policy.storage_admission as policy
    from hephaestus.schemas.storage_admission import LocalStorageProfile
    monkeypatch.setattr(policy,'_mounts',lambda:[('/','overlay','0:27')])
    admitted=policy.admit_storage(tmp_path/'state.sqlite3')
    assert admitted.retention_scope=='container/host-lifetime-only'
    assert not admitted.cross_host_supported and not admitted.external_rollback_protection
    with pytest.raises(PermissionError,match='filesystem'):
        policy.admit_storage(Path('/var/lib/hephaestus/test/state.sqlite3'),
            LocalStorageProfile('/var/lib/hephaestus','persistent_research'))


def test_mutable_repository_read_methods_do_not_change_a_delete_journal_database(tmp_path):
    import hashlib
    path=tmp_path/'read-inspector.sqlite3';repository=SQLiteStateRepository(path)
    repository.append('probe',{'id':'before'})
    with closing(sqlite3.connect(path)) as c:
        assert c.execute('PRAGMA journal_mode=DELETE').fetchone()[0]=='delete'
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    inspector=SQLiteStateRepository(path,initialize=False)
    assert inspector.all('probe')==[{'id':'before'}]
    assert inspector.schema_version()==1
    inspector.assert_integrity()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_inplace_foreign_database_replacement_is_detected_by_logical_identity(tmp_path):
    import hashlib
    path=tmp_path/'primary.sqlite3';repository=SQLiteStateRepository(path)
    repository.append('probe',{'id':'original'})
    foreign=SQLiteStateRepository(tmp_path/'foreign.sqlite3');foreign.append('probe',{'id':'foreign'})
    backup=tmp_path/'foreign-backup.sqlite3';foreign.snapshot_to(backup)
    inode=path.stat().st_ino
    # Explicit disposable offline corruption fixture; no SQLite handles are open.
    with path.open('wb') as stream:stream.write(backup.read_bytes())
    assert path.stat().st_ino==inode
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(StateIntegrityError,match='logical database identity'):
        repository.append('probe',{'invalid':'must not execute'})
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_unrecognized_schema_version_is_not_silently_downgraded(tmp_path):
    import hashlib
    path=tmp_path/'future-version.sqlite3';SQLiteStateRepository(path)
    with closing(sqlite3.connect(path)) as c:
        c.execute("UPDATE infrastructure_schema_migrations SET version=99 WHERE component='state_repository'")
        c.commit()
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(StateIntegrityError,match='schema version'):
        SQLiteStateRepository(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def _fresh_process_writer(path,barrier,worker):
    barrier.wait(timeout=20)
    repository=SQLiteStateRepository(Path(path))
    for i in range(30):
        repository.append('concurrent',{'worker':worker,'i':i},operation_id=f'worker-{worker}-{i}')


def test_fresh_process_initializers_and_writers_share_one_database_identity(tmp_path):
    import multiprocessing
    context=multiprocessing.get_context('spawn');barrier=context.Barrier(4)
    path=tmp_path/'fresh-concurrent.sqlite3'
    children=[context.Process(target=_fresh_process_writer,args=(str(path),barrier,n)) for n in range(4)]
    for child in children:child.start()
    for child in children:
        child.join(timeout=30)
        assert child.exitcode==0
    repository=SQLiteStateRepository(path,read_only=True)
    repository.assert_integrity()
    assert len(repository.all('concurrent'))==120
    assert repository.storage_identity()['database_id']


def test_profile_bound_artifact_store_rechecks_path_aliases_at_write(tmp_path):
    from hephaestus.storage.filesystem import FileSystemArtifactStore
    from hephaestus.schemas.storage_admission import LocalStorageProfile
    root=tmp_path/'artifact-root';root.mkdir()
    foreign=tmp_path/'foreign-root';foreign.mkdir()
    store=FileSystemArtifactStore(root,storage_profile=LocalStorageProfile())
    root.rmdir();root.symlink_to(foreign,target_is_directory=True)
    with pytest.raises(PermissionError,match='symlink'):
        store.put_bytes(b'must not write through a replaced artifact root')
    assert list(foreign.iterdir())==[]


def test_job_queue_reads_close_descriptors_without_garbage_collection(tmp_path):
    import gc
    from hephaestus.jobs import SQLiteJobQueue
    queue=SQLiteJobQueue(tmp_path/'jobs.sqlite3');job=queue.submit('payload','owner',idempotency_key='one')
    gc.collect();enabled=gc.isenabled();gc.disable();before=len(os.listdir('/proc/self/fd'))
    try:
        for _ in range(200):
            assert queue.get(job.job_id).job_id==job.job_id
            assert queue.schema_version()==1
        assert len(os.listdir('/proc/self/fd'))<=before+3
    finally:
        if enabled:gc.enable()
        gc.collect()


def test_job_queue_commit_acknowledgement_loss_keeps_job_identity(tmp_path,monkeypatch):
    from hephaestus.jobs import SQLiteJobQueue
    queue=SQLiteJobQueue(tmp_path/'jobs-uncertain.sqlite3');original=queue._db.connect
    class LostAcknowledgement:
        def __init__(self,connection):self.connection=connection
        def __getattr__(self,name):return getattr(self.connection,name)
        def commit(self):
            self.connection.commit()
            raise sqlite3.OperationalError('lost acknowledgement after actual queued-job commit')
    with monkeypatch.context() as patch:
        patch.setattr(queue._db,'connect',lambda *a,**kw:LostAcknowledgement(original(*a,**kw)))
        with pytest.raises(RuntimeError,match='unknown') as error:
            queue.submit('payload','owner',idempotency_key='stable-job')
    assert len(error.value.operation_ids)==1
    job=queue.get(error.value.operation_ids[0])
    assert job is not None and queue.submit('payload','owner',idempotency_key='stable-job').job_id==job.job_id


def test_corruption_after_construction_blocks_execution_ownership_before_invocation(tmp_path):
    import hashlib
    from hephaestus.state.execution_ownership import execution_owner
    path=tmp_path/'bad-after-construction.sqlite3';repository=SQLiteStateRepository(path)
    repository.append('probe',{'id':'one'})
    with closing(sqlite3.connect(path)) as c:
        page=c.execute("select rootpage from sqlite_master where name='state_latest_key'").fetchone()[0]
        size=c.execute('PRAGMA page_size').fetchone()[0]
        c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    with path.open('r+b') as stream:stream.seek((page-1)*size);stream.write(b'\x7f')
    before=hashlib.sha256(path.read_bytes()).hexdigest();invocations=[]
    with pytest.raises(StateIntegrityError):
        with execution_owner(repository,'workflow','no-side-effects'):
            invocations.append('must not execute')
    assert invocations==[] and hashlib.sha256(path.read_bytes()).hexdigest()==before


@pytest.mark.parametrize('boundary',['postcommit_identity_read','postcommit_close'])
def test_postcommit_io_failures_are_unknown_not_retriable_rollbacks(tmp_path,monkeypatch,boundary):
    path=tmp_path/'postcommit.sqlite3';repository=SQLiteStateRepository(path);original=repository._db.connect
    class PostCommitFailure:
        committed=False
        def __init__(self,connection):self.connection=connection
        def __getattr__(self,name):return getattr(self.connection,name)
        def commit(self):self.connection.commit();self.committed=True
        def execute(self,sql,*args):
            if self.committed and boundary=='postcommit_identity_read':
                raise sqlite3.OperationalError('injected postcommit metadata read failure')
            return self.connection.execute(sql,*args)
        def close(self):
            self.connection.close()
            if self.committed and boundary=='postcommit_close':
                raise sqlite3.OperationalError('injected postcommit close failure')
    with monkeypatch.context() as patch:
        patch.setattr(repository._db,'connect',lambda *a,**kw:PostCommitFailure(original(*a,**kw)))
        with pytest.raises(RuntimeError,match='unknown') as error:
            repository.append('publications',{'id':'one'})
    assert len(error.value.operation_ids)==1
    reopened=SQLiteStateRepository(path)
    reopened.append('publications',{'id':'one'},operation_id=error.value.operation_ids[0])
    assert len(reopened.all('publications'))==1
