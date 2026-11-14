"""Actual CLI intake only, does not execute model or launch training."""
import json

from hephaestus.research.cli import main
from hephaestus.jobs.sqlite import SQLiteJobQueue
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import read_verified_json


def test_cli_enqueues_supported_objective_without_starting_any_compute(tmp_path,capsys):
    root=tmp_path/"durable"
    args=["--state-root",str(root),"submit","--objective",
          "Investigate Diagnostician reliability and evidence coverage",
          "--run-id","cli-research","--lineage-id","cli-lineage"]
    assert main(args)==0
    output=json.loads(capsys.readouterr().out)
    assert output["execution_started"] is False
    assert output["training_authorized"] is False
    assert output["automatic_merge_authorized"] is False
    assert output["status"]=="queued"
    queued=SQLiteJobQueue(root/"state.db").get(output["job_id"])
    assert queued is not None
    assert read_verified_json(FileSystemArtifactStore(root/"artifacts"),
                              queued.payload_ref)["kind"]=="diagnosis_evaluation_only"
    assert main(args)==0
    repeated=json.loads(capsys.readouterr().out)
    assert repeated["job_id"]==output["job_id"]


def test_cli_rejects_non_absolute_state_location_without_work(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        main(["--state-root","relative","submit",
              "--objective","Investigate Diagnostician reliability and evidence coverage",
              "--run-id","x","--lineage-id","y"])
