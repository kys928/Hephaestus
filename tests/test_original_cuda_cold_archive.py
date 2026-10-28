"""No fake CUDA campaign can pass the independent original cold verifier."""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
import json

import pytest
from hephaestus.storage.verified import canonical_bytes

import importlib.util
ROOT=Path(__file__).resolve().parents[1]

def _load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

freeze=_load_script("freeze_original_tier_a_cuda_archive").freeze
verify_original_cuda_archive=_load_script("cold_verify_original_tier_a_cuda_archive").verify_original_cuda_archive


def test_cold_checker_refuses_empty_or_fabricated_45(tmp_path):
    archive=tmp_path/"fake.zip"
    with ZipFile(archive,"w") as z:
        z.writestr("archive_sha256.json","{}")
        z.writestr("state.sqlite",b"not a real 45-operation database")
    sha="sha256:"+sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(PermissionError):
        verify_original_cuda_archive(archive,expected_sha256=sha)


def test_cold_checker_requires_separately_pinned_archive_sha(tmp_path):
    archive=tmp_path/"a.zip"
    with ZipFile(archive,"w") as z:
        z.writestr("archive_sha256.json","{}")
    with pytest.raises(PermissionError,match="independently pinned"):
        verify_original_cuda_archive(archive,expected_sha256="sha256:"+"0"*64)


def test_freeze_refuses_claimed_success_without_actual_complete_science(tmp_path):
    root=tmp_path/"unverified-original"
    root.mkdir()
    fake={"status":"completed","completed_operations":45,
          "verified":{"native_operations":43,"actual_cuda_optimizer_steps":96}}
    from hashlib import sha256
    raw=canonical_bytes(fake)
    (root/("cuda-native-report-"+sha256(raw).hexdigest()+".json")).write_bytes(raw)
    with pytest.raises(PermissionError,match="exactly one"):
        freeze(root,tmp_path/"release.zip")


def test_freeze_never_overwrites_existing_original_archive(tmp_path):
    root=tmp_path/"workflow"
    root.mkdir()
    archive=tmp_path/"archive.zip"
    archive.write_bytes(b"existing evidence")
    with pytest.raises(PermissionError,match="fresh archive"):
        freeze(root,archive)
    assert archive.read_bytes()==b"existing evidence"
