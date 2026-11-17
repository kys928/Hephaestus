"""Real worker deployment never silently falls back to CPU or self-authorization."""
from __future__ import annotations

import json
import pytest

from hephaestus.research.deploy_worker import (
    DeployedResearchWorkerScope,compose_live_queue_worker,
)
from hephaestus.production.original_cuda_identity import registry_digest


def _scope(tmp_path,**changes):
    paths={
        "state_root":tmp_path/"state",
        "run_id":"research-run","lineage_id":"lineage",
        "source_catalog_path":tmp_path/"catalog.json",
        "pinned_source_catalog_sha256":"sha256:"+"a"*64,
        "signed_grant_path":tmp_path/"external-grant.json",
        "trusted_public_key_path":tmp_path/"public-key.txt",
        "pinned_public_key_sha256":"sha256:"+"b"*64,
        "trusted_role_registry_sha256":registry_digest(),
        "eval_pack_path":tmp_path/"pack.json",
        "original_model_cache":tmp_path/"model-cache",
        "original_adapter_cache":tmp_path/"adapter-cache",
        "s3_endpoint":"https://s3api-eu-cz-1.runpod.io/",
        "s3_region":"EU-CZ-1",
    }
    paths.update(changes)
    return DeployedResearchWorkerScope(**paths)


def test_missing_operator_grant_fails_before_queue_model_or_s3_use(tmp_path):
    scope=_scope(tmp_path)
    with pytest.raises(PermissionError,match="missing required trusted policy"):
        compose_live_queue_worker(scope)
    assert not scope.state_root.exists()


@pytest.mark.parametrize("invalid",[
    {"s3_region":"US-XX-1"},{"s3_endpoint":"https://attacker.invalid/"},
    {"run_id":"../../bad"},{"trusted_role_registry_sha256":"sha256:"+"0"*64},
    {"worker_id":"bad/worker"},
])
def test_external_scope_drift_not_accepted(tmp_path,invalid):
    with pytest.raises(PermissionError):
        _scope(tmp_path,**invalid).validate()


def test_cli_signed_worker_does_not_allow_empty_external_authority(tmp_path):
    from hephaestus.research.cli import main
    with pytest.raises(SystemExit):
        main(["--state-root",str(tmp_path/"durable"),"serve-signed-cuda"])
