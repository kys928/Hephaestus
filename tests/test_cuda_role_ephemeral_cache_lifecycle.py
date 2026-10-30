"""Offline real-cache lifecycle checks; never download weights or rent GPU."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import pytest

from hephaestus.llm.certified_runtime import PinnedTransformersRoleGenerator


def generator(root):
    return PinnedTransformersRoleGenerator(
        hf_cache_dir=root,allow_remote_pinned_download=True)


def old_identity(role="controller",model="Qwen/Qwen3-4B-Instruct-2507"):
    return (role,model,"a"*40,"b"*64,"/tmp/dummy-adapter")


def test_retired_cache_evicts_only_one_completed_exact_model(tmp_path):
    root=tmp_path/"original-role-cache"
    old=root/"models--Qwen--Qwen3-4B-Instruct-2507"
    next_role=root/"models--mistralai--Ministral-3-3B-Reasoning-2512"
    old_snapshot=old/"blobs"/"immutable-test"
    next_snapshot=next_role/"blobs"/"keep-test"
    old_snapshot.parent.mkdir(parents=True)
    next_snapshot.parent.mkdir(parents=True)
    old_snapshot.write_text("old-role-temporary-test-weight")
    next_snapshot.write_text("new-role-temporary-test-weight")
    unrelated=root/"artifacts-test.txt"
    unrelated.write_text("keep")
    generator(root)._evict_retired_role_cache(old_identity())
    assert not old.exists()
    assert next_snapshot.read_text()=="new-role-temporary-test-weight"
    assert unrelated.read_text()=="keep"


def test_retired_cache_refuses_untrusted_outside_tmp_location():
    outside=Path("/workspace/hephaestus/scientific/original_cuda_45")
    with pytest.raises(PermissionError,match="temporary storage"):
        generator(outside)._evict_retired_role_cache(old_identity())


def test_retired_cache_rejects_symlink_root_and_model_cache(tmp_path):
    real=tmp_path/"real"
    real.mkdir()
    link=tmp_path/"link"
    link.symlink_to(real,target_is_directory=True)
    with pytest.raises(PermissionError,match="unsafe"):
        generator(link)._evict_retired_role_cache(old_identity())

    normal=tmp_path/"good"
    normal.mkdir()
    target=tmp_path/"target"
    target.mkdir()
    (normal/"models--Qwen--Qwen3-4B-Instruct-2507").symlink_to(target,target_is_directory=True)
    with pytest.raises(PermissionError,match="symlinks"):
        generator(normal)._evict_retired_role_cache(old_identity())
    assert target.exists()


def test_retired_cache_refuses_traversal_or_foreign_model_identity(tmp_path):
    root=tmp_path/"cache"
    root.mkdir()
    for model in ("../../workspace","Qwen/../bad","/absolute","bad","Qwen/path/extra"):
        with pytest.raises(PermissionError,match="unsafe"):
            generator(root)._evict_retired_role_cache(old_identity(model=model))


def test_disk_capacity_check_fails_before_download_and_handles_nonexistent_cache(tmp_path,monkeypatch):
    root=tmp_path/"not-created-yet"
    gen=generator(root)
    disk_calls=[]
    def fake_disk_usage(path):
        disk_calls.append(Path(path))
        return SimpleNamespace(free=21*1024**3)
    monkeypatch.setattr("hephaestus.llm.certified_runtime.shutil.disk_usage",fake_disk_usage)
    with pytest.raises(RuntimeError,match="insufficient free disk"):
        gen._ensure_role_cache_capacity()
    assert disk_calls and disk_calls[0].exists()
    monkeypatch.setattr("hephaestus.llm.certified_runtime.shutil.disk_usage",
                        lambda path:SimpleNamespace(free=23*1024**3))
    gen._ensure_role_cache_capacity()


def test_offline_read_only_inference_never_evicts_models(tmp_path):
    root=tmp_path/"cache"
    old=root/"models--Qwen--Qwen3-4B-Instruct-2507"
    old.mkdir(parents=True)
    gen=PinnedTransformersRoleGenerator(hf_cache_dir=root,
                                        allow_remote_pinned_download=False)
    gen._evict_retired_role_cache(old_identity())
    assert old.exists()
