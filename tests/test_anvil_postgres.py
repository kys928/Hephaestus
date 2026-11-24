"""Database-backed Anvil contract qualification, opt in with disposable service DSNs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
import subprocess
import sys

import pytest

from hephaestus.schemas.anvil import Contribution, ContributionKind, ContributionOrigin
from hephaestus.state.anvil_postgres import PostgresAnvilState, SCHEMA, advance_epoch


def test_postgres_anvil_admission_and_independent_database_restore(tmp_path):
    dsn = os.environ.get("ANVIL_ADAPTER_SOURCE_DSN")
    if not dsn:
        pytest.skip("disposable PostgreSQL adapter services are not configured")
    import psycopg

    restored_dsn = os.environ["ANVIL_ADAPTER_RESTORE_DSN"]
    source = os.environ["ANVIL_ADAPTER_SOURCE_CONTAINER"]
    destination = os.environ["ANVIL_ADAPTER_RESTORE_CONTAINER"]
    with psycopg.connect(dsn) as connection:
        connection.execute(SCHEMA)
        assert advance_epoch(connection, "study", expected_epoch=0) == 1
        connection.execute("CREATE ROLE anvil_app LOGIN PASSWORD 'fixture'")
        connection.execute("GRANT USAGE ON SCHEMA anvil TO anvil_app")
        connection.execute("GRANT EXECUTE ON FUNCTION anvil.lock_epoch(text) TO anvil_app")
        connection.execute("GRANT SELECT, INSERT ON anvil.contribution TO anvil_app")
    app_dsn = psycopg.conninfo.make_conninfo(dsn, user="anvil_app", password="fixture")

    def board_for(connection_string, epoch):
        return PostgresAnvilState(lambda: psycopg.connect(connection_string), "study", epoch,
                                  lambda _: True, lambda *_: False)

    board = board_for(app_dsn, 1)
    empty = board.read_snapshot("study")
    origin = ContributionOrigin("study", "run", "lineage", "diagnosis", "sha256:" + "a" * 64,
                                empty.frontier, "entry-0", "org/model", "0" * 40, "sha256:" + "f" * 64)
    claim = Contribution(ContributionKind.HYPOTHESIS, "A fixture claim")

    def submit(index):
        return board.append_contribution(replace(origin, idempotency_key=f"entry-{index}"), claim)

    with ThreadPoolExecutor(max_workers=6) as pool:
        accepted = list(pool.map(submit, [0, 1, 2, 3, 4, 0]))
    assert accepted[0] == accepted[-1]
    assert len({item.frontier for item in accepted}) == 5
    with pytest.raises(ValueError, match="conflicting"):
        board.append_contribution(origin, replace(claim, text="conflicting retry"))

    # Kill after a real adapter insert but before the outer transaction commits.
    from hephaestus.state.anvil import _collection

    interrupted = """
import os
import psycopg
from hephaestus.state.anvil_postgres import PostgresAnvilState
board = PostgresAnvilState(lambda: psycopg.connect(os.environ['ANVIL_CHILD_DSN']), 'study', 1,
                          lambda _: True, lambda *_: False)
with board._write(os.environ['ANVIL_CHILD_COLLECTION'], 'study') as (records, writer):
    writer.append(os.environ['ANVIL_CHILD_COLLECTION'], {'sequence': len(records) + 1},
                  operation_id='interrupted')
    os._exit(37)
"""
    env = dict(os.environ, ANVIL_CHILD_DSN=app_dsn, ANVIL_CHILD_COLLECTION=_collection("study"))
    assert subprocess.run([sys.executable, "-c", interrupted], env=env, check=False).returncode == 37
    with psycopg.connect(dsn) as connection:
        assert advance_epoch(connection, "study", expected_epoch=1) == 2
    with pytest.raises(PermissionError, match="stale"):
        submit(9)
    with pytest.raises(psycopg.errors.InsufficientPrivilege), psycopg.connect(app_dsn) as connection:
        advance_epoch(connection, "study", expected_epoch=2)
    replacement = board_for(app_dsn, 2)
    last = replacement.append_contribution(replace(origin, idempotency_key="replacement"), claim)
    snapshot = replacement.replay("study", expected_frontier=last.frontier)
    assert len(snapshot.contributions) == 6

    backup = tmp_path / "anvil.sql"
    with backup.open("wb") as stream:
        subprocess.run(["docker", "exec", source, "pg_dump", "-U", "postgres", "-n", "anvil",
                        "--no-acl", "--no-owner", "anvil_fixture"], stdout=stream, check=True)
    with backup.open("rb") as stream:
        subprocess.run(["docker", "exec", "-i", destination, "psql", "-U", "postgres",
                        "-v", "ON_ERROR_STOP=1", "anvil_fixture"], stdin=stream, check=True)
    restored = board_for(restored_dsn, 2)
    assert restored.replay("study", expected_frontier=last.frontier) == snapshot
    assert restored.reconcile("study", "entry-0") == accepted[0]
    with pytest.raises(ValueError, match="retained frontier"):
        restored.replay("study", expected_frontier=empty.frontier)
