"""Real storage publication must finish file and directory barriers before acknowledgement."""
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from hephaestus.production.state import RepositoryIntegrationRecordSink, INTEGRATION_RECORDS
from hephaestus.state._json_store import JsonSingleDocument, JsonStore
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.training.hf_worker import _write_json


@pytest.mark.parametrize("kind", ["artifact", "lineage", "worker_metadata", "legacy_history"])
def test_actual_publications_flush_directory_before_return(tmp_path, monkeypatch, kind):
    synced = []
    original = os.fsync

    def track(fd):
        synced.append(("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file",
                       os.readlink(f"/proc/self/fd/{fd}")))
        return original(fd)

    monkeypatch.setattr(os, "fsync", track)
    if kind == "artifact":
        FileSystemArtifactStore(tmp_path / "artifacts").put_bytes(b"durable scientific bytes")
    elif kind == "lineage":
        JsonSingleDocument(tmp_path / "state", "lineage_states.json").write({"lineage": "accepted"})
    elif kind == "worker_metadata":
        _write_json(tmp_path / "worker" / "final_result.json", {"status": "completed"})
    else:
        JsonStore(tmp_path / "state", "runs.jsonl").append({"run_id": "real-state"})
    assert any(kind == "file" for kind, _ in synced), synced
    assert synced[-1][0] == "directory", synced


def test_failed_artifact_directory_barrier_cannot_publish_a_state_reference(tmp_path, monkeypatch):
    repository = SQLiteStateRepository(tmp_path / "state.sqlite3")
    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")
    sink = RepositoryIntegrationRecordSink(repository, artifacts)
    original = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected directory sync I/O failure")
        original(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_directory)
        with pytest.raises(OSError, match="directory sync"):
            sink.append("scientific-publication", {"run_id": "one"})
    assert repository.all(INTEGRATION_RECORDS) == []
    # Immutable same-content retry is safe only after its real durability barriers succeed.
    sink.append("scientific-publication", {"run_id": "one"})
    rows = repository.all(INTEGRATION_RECORDS)
    assert len(rows) == 1 and artifacts.verify(rows[0]["artifact_ref"])


def test_killed_real_sqlite_backup_never_leaves_a_completed_destination(tmp_path):
    database = tmp_path / "state.sqlite3"
    repository = SQLiteStateRepository(database)
    for index in range(30):
        repository.append("scientific_records", {"index": index, "bytes": "x" * 6000})
    destination = tmp_path / "completed-backup.sqlite3"
    code = r'''
from contextlib import contextmanager
from pathlib import Path
import os,sys
from hephaestus.storage.sqlite import SQLiteStateRepository
repository=SQLiteStateRepository(Path(sys.argv[1]),read_only=True)
original=repository._db.connection
class InterruptActualBackup:
    def __init__(self,connection): self.connection=connection
    def backup(self,target):
        self.connection.backup(target,pages=1,progress=lambda *_:os._exit(73))
@contextmanager
def interrupted_connection():
    with original() as connection: yield InterruptActualBackup(connection)
repository._db.connection=interrupted_connection
repository.snapshot_to(Path(sys.argv[2]))
'''
    child = subprocess.run([sys.executable, "-c", code, str(database), str(destination)],
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == 73, child.stderr
    assert not destination.exists(), "interrupted backup was exposed under its completed name"
    repository.assert_integrity()
    assert len(repository.all("scientific_records")) == 30


@pytest.mark.real_cpu
def test_actual_optimizer_checkpoint_flushes_all_components_before_directory_publish(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from hephaestus.training import hf_worker
    from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=8, n_positions=16, n_embd=8, n_layer=1, n_head=1))
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"pad":0,"bos":1,"eos":2,"unk":3,"hello":4,"world":5},unk_token="unk")))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    tokens = torch.tensor([[1,4,5,2]])
    loss = model(tokens, labels=tokens).loss
    loss.backward(); optimizer.step(); scheduler.step()
    seen = set(); publication_checks = []
    sync, replace = os.fsync, os.replace

    def track(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            seen.add(os.readlink(f"/proc/self/fd/{fd}"))
        return sync(fd)

    def inspect_publish(source,destination):
        if Path(source).name == "checkpoint_step_1.partial":
            components = {str(p) for p in Path(source).rglob("*") if p.is_file()}
            assert components and components <= seen, components-seen
            assert str(Path(source)/"training_state.pt") in seen
            publication_checks.append(True)
        return replace(source,destination)

    monkeypatch.setattr(os,"fsync",track)
    monkeypatch.setattr(os,"replace",inspect_publish)
    identity = "sha256:"+"a"*64
    compatibility = dict(architecture_family="gpt2", training_mode="full_finetune",
        model_id="authored-tiny-io-test", model_revision=identity, tokenizer_id="authored-byte-fixture",
        tokenizer_revision=identity, processed_dataset_hash=identity)
    checkpoint,_ = hf_worker._save_checkpoint(torch=torch,model=model,tokenizer=tokenizer,
        optimizer=optimizer,scheduler=scheduler,job={"artifact_root":str(tmp_path),
        "run_id":"durability-worker","experiment_id":"durability-io", "config_fingerprint":identity,
        "compatibility":compatibility,"training_config":{"peft_config":None}},step=1,epoch=1.,
        metrics=dict(examples_processed=1,tokens_processed=4,batches_processed=1,
                     training_loss=float(loss.detach()),gradient_norm=0.))
    assert publication_checks == [True]
    assert validate_checkpoint_manifest(checkpoint)[0]
