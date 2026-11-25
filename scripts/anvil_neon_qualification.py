"""Qualify hosted Anvil with a disposable contribution and off-provider restore."""
import argparse
from pathlib import Path
import os

from anvil_neon_bootstrap import connect_runtime, verify_runtime
from hephaestus.schemas.anvil import Contribution, ContributionKind, ContributionOrigin
from hephaestus.state.anvil_postgres import PostgresAnvilState, SCHEMA, advance_epoch
from hephaestus.storage.filesystem import FileSystemArtifactStore

LIVE = "diagnosis-planner-v1"
QUALIFICATION = "diagnosis-planner-neon-qualification-v1"
RECEIPT = "sha256:" + "a" * 64
NOTE = Contribution(ContributionKind.HYPOTHESIS, "Synthetic hosted persistence qualification only.")


def board(connect, epoch):
    return PostgresAnvilState(connect, QUALIFICATION, epoch, lambda origin: origin.inference_ref == RECEIPT,
                              lambda *_: False)


def hosted(artifact_directory: Path) -> None:
    import psycopg

    verify_runtime(LIVE)
    connect = connect_runtime
    active = board(connect, 2)
    origin = ContributionOrigin(
        QUALIFICATION, "qualification", "qualification", "diagnosis", RECEIPT,
        active.read_snapshot(QUALIFICATION, sequence=0).frontier, "hosted-fixture-v1",
        "qualification/model", "0" * 40, "sha256:" + "b" * 64,
    )
    try:
        board(connect, 1).append_contribution(origin, NOTE)
    except PermissionError as error:
        if "stale or unregistered" not in str(error):
            raise
    else:
        raise RuntimeError("stale writer was admitted")
    try:
        with connect() as connection:
            advance_epoch(connection, QUALIFICATION, expected_epoch=2)
    except psycopg.errors.InsufficientPrivilege:
        pass
    else:
        raise RuntimeError("runtime login was able to rotate an epoch")
    accepted = active.append_contribution(origin, NOTE)
    recovered = board(connect, 2)
    if recovered.reconcile(QUALIFICATION, origin.idempotency_key) != accepted:
        raise ValueError("hosted fresh-connection reconciliation differs")
    if recovered.append_contribution(origin, NOTE) != accepted:
        raise ValueError("hosted retry was not idempotent")
    if recovered.replay(QUALIFICATION, expected_frontier=accepted.frontier).contributions != (accepted,):
        raise ValueError("hosted replay differs from the synthetic acceptance")
    artifacts = FileSystemArtifactStore(artifact_directory)
    backup_ref = recovered.checkpoint(QUALIFICATION, artifacts)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        print("backup_ref=" + backup_ref, file=output)
        print("frontier=" + accepted.frontier, file=output)
        print("accepted_id=" + accepted.contribution_id, file=output)
    print("ANVIL_HOSTED_APPEND_REOPEN_RETRY_FENCING_VERIFIED=true")


def restore(artifact_directory: Path) -> None:
    import psycopg

    dsn = os.environ["ANVIL_RESTORE_DSN"]
    connect = lambda: psycopg.connect(dsn, connect_timeout=10)
    with connect() as connection:
        connection.execute(SCHEMA)
        advance_epoch(connection, QUALIFICATION, expected_epoch=0)
    destination = board(connect, 1)
    artifacts = FileSystemArtifactStore(artifact_directory)
    frontier = os.environ["EXPECTED_FRONTIER"]
    destination.restore(QUALIFICATION, artifacts, os.environ["BACKUP_REF"], expected_frontier=frontier)
    reopened = board(connect, 1).replay(QUALIFICATION, expected_frontier=frontier)
    if (len(reopened.contributions) != 1
            or reopened.contributions[0].contribution_id != os.environ["EXPECTED_ACCEPTED_ID"]
            or reopened.contributions[0].payload != NOTE):
        raise ValueError("independent restore differs from the hosted acceptance")
    print("ANVIL_OFF_PROVIDER_RESTORE_VERIFIED=true")
    print("ANVIL_HOSTED_QUALIFICATION_FRONTIER=" + frontier)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["hosted", "restore"])
    parser.add_argument("artifact_directory", type=Path)
    args = parser.parse_args()
    try:
        (hosted if args.mode == "hosted" else restore)(args.artifact_directory)
    except PermissionError as error:
        parser.exit(1, f"Anvil qualification rejected: {error}.\n")
    except Exception as error:
        parser.exit(1, f"Anvil qualification failed ({type(error).__name__}).\n")


if __name__ == "__main__":
    main()
