"""Authentic CPU fixture, complete restore and cold subprocess; no final batch."""
import pytest


@pytest.mark.real_cpu
def test_complete_launch_fixture_preserves_one_use_through_backup_restore_and_cold_process(tmp_path):
    for dependency in ('torch','transformers','peft','cryptography'):pytest.importorskip(dependency)
    from hephaestus.development.prospective_launch_rehearsal import run_launch_rehearsal
    result=run_launch_rehearsal(tmp_path/'campaign',tmp_path/'receipts',tmp_path/'external-fixture',source_commit='0'*40)
    assert result['actual_optimizer_updates']==12
    assert result['complete_snapshot']['retained_checkpoints']==3
    assert result['stale_pre_exposure_restore_denied'] and result['retained_external_fixture_reuse_denied']
    assert result['cold']['model_reexecution_verified'] and result['cold']['token_decoding_verified']
    assert result['missing_external_final_authority_denied_before_material_access']
    assert not result['protected_final_holdout_exposed'] and not result['independent_authority_established']
