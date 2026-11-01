"""Sealed whole-campaign backups at a kernel-fenced, quiescent boundary.

Restore preserves the original absolute root. Moving a bundle is supported;
relocating artifact/checkpoint references is deliberately not implemented.
The separately retained control journal is NEVER copied out of a backup.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from contextlib import ExitStack

from hephaestus.policy.campaign_storage import require_separate_control
from hephaestus.policy.storage_admission import admit_storage, reject_symlinks
from hephaestus.schemas.campaign_storage import CampaignSnapshotManifest
from hephaestus.schemas.storage_admission import LocalStorageProfile
from hephaestus.storage.durability import atomic_write_bytes, sync_tree, sync_directory_chain
from hephaestus.storage.sqlite import SQLiteStateRepository, StateIntegrityError
from hephaestus.storage.verified import canonical_bytes
from .campaign_witness import CampaignWitness, CampaignRollbackError
from .worker_ownership import worker_owner,WORKER_LOCK_FILE
from .single_writer import WriterBusyError
from hephaestus.schemas.campaign_storage import WorkerFenceReference


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _safe_files(root):
    for path in sorted(root.rglob('*')):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise StateIntegrityError('campaign snapshot cannot contain symlinks or special files')
        if path.is_file():
            if path.stat().st_nlink != 1:
                raise StateIntegrityError('campaign files must not have hard-link aliases')
            yield path


def _database(repository):
    repository.assert_integrity()
    return repository.storage_identity()


def _assert_quiescent(root, repository, worker_fences):
    # No controller can start work while the external campaign fence is held.
    # Workers do not take that fence; unknown or live asynchronous jobs block.
    for path in root.rglob('handle.json'):
        handle = json.loads(path.read_text())
        if handle.get('status') not in {'completed', 'failed', 'cancelled', 'interrupted'}:
            raise StateIntegrityError('campaign snapshot requires every training handle to be settled')
        pid = handle.get('metadata', {}).get('pid')
        metadata=handle.get('metadata',{})
        descriptor=metadata.get('worker_fence')
        if descriptor is not None:
            reject_symlinks(path.parent/WORKER_LOCK_FILE)
            if descriptor!=WorkerFenceReference().to_dict() or not (path.parent/WORKER_LOCK_FILE).is_file():
                raise StateIntegrityError('worker fence is missing or unsupported; no automatic replacement proof')
            if (path.parent/WORKER_LOCK_FILE).stat().st_nlink!=1:
                raise StateIntegrityError('worker fence must not have hard-link aliases')
            try:
                worker_fences.enter_context(worker_owner(path.parent,prove_quiescent=True))
            except WriterBusyError as exc:
                raise StateIntegrityError('worker still owns its producer fence; snapshot must wait for quiescence') from exc
        if isinstance(pid, int):
            # Bare PIDs can be reused and refer to different processes in a new
            # PID namespace. Accept a reaped exit or a bound terminal producer
            # result whose worker fence is unowned and held through the copy.
            if type(metadata.get('returncode')) is not int:
                if descriptor is None:
                    raise StateIntegrityError('worker exit is unacknowledged; reconcile/reap the original worker before snapshot')
                try:
                    result=json.loads((path.parent/'runtime_result.json').read_text())
                except (OSError,ValueError) as exc:
                    raise StateIntegrityError('recovered worker lacks a readable terminal result') from exc
                if (result.get('status')!=handle.get('status') or result.get('status') not in {'completed','interrupted','cancelled'}
                        or result.get('run_id')!=handle.get('run_id') or not metadata.get('config_fingerprint')
                        or result.get('config_fingerprint')!=metadata.get('config_fingerprint')):
                    raise StateIntegrityError('recovered terminal worker identity does not match its fenced result')
    latest = {}
    for row in repository.all('production_action_executions'):
        latest[row['execution_id']] = row
    if any(row.get('status') == 'prepared' for row in latest.values()):
        raise StateIntegrityError('reconcile prepared cross-store actions before snapshot')
    # A consumed assessment is preserved, but no in-flight generation may be
    # snapshotted as a resumable permission.
    sessions = {r['study_id']: r for r in repository.all('prospective_custody_sessions')}
    if any(r.get('status') == 'consumed' for r in sessions.values()):
        raise StateIntegrityError('in-flight exposure blocks snapshot; it must remain permanently consumed')


def _verify_relationships(root, repository, *, logical_root=None):
    """Check content addresses and every retained checkpoint reference."""
    logical_root = Path(logical_root or root)
    objects = {}
    for path in _safe_files(root):
        if path.parent.parent.name == 'sha256' and path.parent.parent.parent.name == 'objects':
            if len(path.name) != 64 or _hash(path) != path.name:
                raise StateIntegrityError('content-addressed campaign artifact is corrupt')
            objects[path.name] = path
    with repository._db.connection() as connection:
        payloads = [str(row[0]) for row in connection.execute('SELECT payload_json FROM state_records ORDER BY sequence')]
    # A checkpoint manifest/optimizer/model digest is not necessarily an object
    # in the content-addressed store. Follow typed reference fields, preserving
    # these distinct identities rather than treating every digest as an object.
    identity_refs = {'baseline_ref', 'checkpoint_ref', 'model_ref', 'tokenizer_ref', 'parent_adapter_ref'}
    def refs(value, key=''):
        if isinstance(value, dict):
            for name, item in value.items():
                yield from refs(item, key if key.endswith('_refs') else name)
        elif isinstance(value, list):
            for item in value:
                yield from refs(item, key)
        elif (isinstance(value, str) and value.startswith('sha256:')
                and (key == 'ref' or key.endswith(('_ref', '_refs')))
                and key not in identity_refs):
            yield value[7:]
    for payload in payloads:
        if any(ref not in objects for ref in refs(json.loads(payload))):
            raise StateIntegrityError('committed record references an absent campaign artifact')
    from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
    for row in repository.all('checkpoint_identities'):
        original = reject_symlinks(Path(row['checkpoint_ref']))
        if not original.is_relative_to(logical_root):
            raise StateIntegrityError('checkpoint outside campaign root must be brought into the campaign before snapshot')
        checkpoint = root / original.relative_to(logical_root)
        valid, identity = validate_checkpoint_manifest(checkpoint)
        if not valid or identity != row['manifest_hash']:
            raise StateIntegrityError('campaign checkpoint differs from retained producer identity')
    return {'state_records': len(payloads), 'content_addressed_objects': len(objects),
            'retained_checkpoints': len(repository.all('checkpoint_identities'))}


def snapshot_campaign(root, repository, destination):
    root, destination = reject_symlinks(root), reject_symlinks(destination)
    if not repository.path.is_relative_to(root):
        raise ValueError('state database must belong to the complete campaign root')
    profile = repository._db.profile
    admit_storage(root, profile)
    admit_storage(destination, profile)
    witness = repository._db.witness()
    if witness is None:
        raise CampaignRollbackError('whole-campaign snapshot requires an enrolled retained witness')
    require_separate_control(root, witness.root)
    require_separate_control(destination, witness.root)
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError('backup and campaign roots must be disjoint')
    with witness.owner(blocking=False), ExitStack() as worker_fences:
        witness.verify_chain()
        identity = _database(repository)
        witness.assert_current(identity, required=True, reconcile=False)
        _assert_quiescent(root, repository,worker_fences)
        relationships = _verify_relationships(root, repository)
        if destination.exists():
            raise FileExistsError('completed campaign snapshot destination must be new')
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.', suffix='.partial', dir=destination.parent))
        payload = partial / 'payload'
        payload.mkdir()
        database_relative = repository.path.relative_to(root)
        excluded = {repository.path, Path(str(repository.path)+'-wal'), Path(str(repository.path)+'-shm')}
        for path in _safe_files(root):
            if path in excluded or (path.name.endswith('-writer.lock') and path.name!=WORKER_LOCK_FILE) or path.name.endswith('.initialization.lock'):
                continue
            if path.name.endswith(('.sqlite', '.sqlite3', '.db')):
                raise StateIntegrityError('additional SQLite database requires its own coordinated backup; unsupported')
            if path.name.endswith('.partial') or any(p.endswith('.partial') for p in path.relative_to(root).parts):
                raise StateIntegrityError('unresolved partial publication blocks completed snapshot')
            target = payload / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        repository.snapshot_to(payload / database_relative)
        # Every SQL reader is closed before hashing the private backup. Never
        # open/close raw descriptors for the live database while SQLite uses it.
        files = {p.relative_to(payload).as_posix(): {'sha256': _hash(p), 'bytes': p.stat().st_size}
                 for p in _safe_files(payload)}
        manifest = CampaignSnapshotManifest(str(root), database_relative.as_posix(), identity, witness.receipt(), files).to_dict()
        manifest['relationships'] = relationships
        manifest_bytes = canonical_bytes(manifest)
        atomic_write_bytes(partial / 'manifest.json', manifest_bytes, overwrite=False)
        atomic_write_bytes(partial / 'seal.json', canonical_bytes({'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}), overwrite=False)
        sync_tree(partial)
        os.rename(partial, destination)
        sync_directory_chain(destination.parent)
        return {'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'snapshot': str(destination),
                'storage_identity': identity, 'witness': witness.receipt(), **relationships}


def verify_snapshot(snapshot, *, expected_manifest_sha256):
    snapshot = reject_symlinks(snapshot)
    manifest_bytes = (snapshot / 'manifest.json').read_bytes()
    if (_hash(snapshot / 'manifest.json') != expected_manifest_sha256
            or json.loads((snapshot / 'seal.json').read_text()) != {'manifest_sha256': expected_manifest_sha256}):
        raise StateIntegrityError('snapshot differs from separately retained manifest commitment')
    manifest = json.loads(manifest_bytes)
    if (manifest['schema'] != 'campaign-snapshot.v1' or manifest['relocatable'] is not False
            or manifest['independent_exposure_authority'] is not False):
        raise StateIntegrityError('unsupported campaign snapshot scope')
    payload = snapshot / 'payload'
    observed = {p.relative_to(payload).as_posix(): {'sha256': _hash(p), 'bytes': p.stat().st_size}
                for p in _safe_files(payload)}
    if observed != manifest['files']:
        raise StateIntegrityError('snapshot file set, size or content differs')
    database = payload / manifest['database_relative_path']
    if not database.is_relative_to(payload) or '..' in database.parts:
        raise StateIntegrityError('invalid snapshot database path')
    repository = SQLiteStateRepository(database, read_only=True)
    if _database(repository) != manifest['storage_identity']:
        raise StateIntegrityError('snapshot SQL commit identity differs')
    relationships = _verify_relationships(payload, repository, logical_root=manifest['logical_root'])
    if relationships != manifest['relationships']:
        raise StateIntegrityError('snapshot artifact/checkpoint relationships differ')
    return manifest


def restore_campaign(snapshot, target, *, expected_manifest_sha256, storage_profile=None,
                     quarantine_existing_to=None):
    target = reject_symlinks(target)
    profile = storage_profile or LocalStorageProfile()
    admit_storage(target, profile)
    manifest = verify_snapshot(snapshot, expected_manifest_sha256=expected_manifest_sha256)
    if str(target) != manifest['logical_root']:
        raise ValueError('absolute-path campaign must be restored at its original logical root; relocation is unsupported')
    witness = CampaignWitness(target / manifest['database_relative_path'], profile)
    require_separate_control(target, witness.root)
    with witness.owner(blocking=False), ExitStack() as worker_fences:
        witness.verify_chain()
        witness.assert_current(manifest['storage_identity'], required=True, reconcile=False)
        if witness.receipt() != manifest['witness']:
            raise CampaignRollbackError('snapshot is not the latest separately retained campaign checkpoint')
        if target.exists():
            if quarantine_existing_to is None:
                raise FileExistsError('restore never replaces an existing campaign; supply a new quarantine destination')
            quarantine = reject_symlinks(quarantine_existing_to)
            admit_storage(quarantine, profile)
            require_separate_control(quarantine, witness.root)
            snapshot_path = reject_symlinks(snapshot)
            if (quarantine.exists() or quarantine.is_relative_to(target)
                    or target.is_relative_to(quarantine) or quarantine.is_relative_to(snapshot_path)
                    or snapshot_path.is_relative_to(quarantine)):
                raise ValueError('quarantine must be a new disjoint directory')
            live = SQLiteStateRepository(target / manifest['database_relative_path'], read_only=True)
            if _database(live) != manifest['storage_identity']:
                raise CampaignRollbackError('live state advanced or diverged since snapshot; no replacement permitted')
            _assert_quiescent(target, live,worker_fences)
            os.rename(target, quarantine)
            sync_directory_chain(quarantine.parent)
            sync_directory_chain(target.parent)
        partial = Path(tempfile.mkdtemp(prefix=f'.{target.name}.restore.', suffix='.partial', dir=target.parent))
        shutil.copytree(Path(snapshot) / 'payload', partial, dirs_exist_ok=True)
        # Validate the actual installed candidate, not only the source bundle.
        for relative, expected in manifest['files'].items():
            path = partial / relative
            if path.stat().st_size != expected['bytes'] or _hash(path) != expected['sha256']:
                raise StateIntegrityError('restore copy differs from sealed source')
        sync_tree(partial)
        os.rename(partial, target)
        sync_directory_chain(target.parent)
        return manifest
