"""Fenced PostgreSQL implementation of the Anvil contribution ledger."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, ContextManager

from hephaestus.schemas.anvil import ContributionOrigin
from hephaestus.state.anvil import AnvilState, _collection


SCHEMA = """
CREATE SCHEMA anvil;
CREATE TABLE anvil.writer (
    collection text PRIMARY KEY,
    epoch bigint NOT NULL CHECK (epoch > 0)
);
CREATE TABLE anvil.contribution (
    collection text NOT NULL REFERENCES anvil.writer(collection),
    sequence bigint NOT NULL CHECK (sequence > 0),
    identity text NOT NULL,
    document jsonb NOT NULL,
    PRIMARY KEY (collection, sequence),
    UNIQUE (collection, identity)
);
CREATE FUNCTION anvil.lock_epoch(target text) RETURNS bigint
LANGUAGE sql SECURITY DEFINER SET search_path = pg_catalog
AS $$ SELECT epoch FROM anvil.writer WHERE collection = target FOR UPDATE $$;
REVOKE ALL ON FUNCTION anvil.lock_epoch(text) FROM PUBLIC;
"""


@dataclass(slots=True)
class _TransactionWriter:
    connection: object

    def append(self, collection: str, record: dict, *, operation_id: str) -> None:
        from psycopg.types.json import Jsonb

        self.connection.execute(
            "INSERT INTO anvil.contribution (collection, sequence, identity, document) VALUES (%s, %s, %s, %s)",
            (collection, record["sequence"], operation_id, Jsonb(record)),
        )


class PostgresAnvilState(AnvilState):
    """Use a controller-issued generation scoped to one investigation.

    The connection factory must return a new non-autocommit connection. A commit
    whose acknowledgement is lost can be reconciled by contribution identity.
    """

    def __init__(
        self, connect: Callable[[], ContextManager], investigation: str, epoch: int,
        verify_origin: Callable[[ContributionOrigin], bool],
        verify_evidence: Callable[[ContributionOrigin, str], bool],
    ) -> None:
        if type(epoch) is not int or epoch < 1:
            raise ValueError("a positive writer epoch is required")
        self._connect = connect
        self._collection = _collection(investigation)
        self._epoch = epoch
        self.verify_origin = verify_origin
        self.verify_evidence = verify_evidence

    def _check_scope(self, collection: str) -> None:
        if collection != self._collection:
            raise PermissionError("PostgreSQL Anvil binding is scoped to another investigation")

    @staticmethod
    def _read(connection, collection: str) -> list[dict[str, object]]:
        return [row[0] for row in connection.execute(
            "SELECT document FROM anvil.contribution WHERE collection = %s ORDER BY sequence", (collection,),
        )]

    def _records(self, collection: str) -> list[dict[str, object]]:
        self._check_scope(collection)
        with self._connect() as connection:
            return self._read(connection, collection)

    @contextmanager
    def _write(self, collection: str, investigation: str, *, atomic_batch: bool = False):
        self._check_scope(collection)
        with self._connect() as connection:
            if connection.autocommit:
                raise ValueError("Anvil writes require transactional connections")
            with connection.transaction():
                row = connection.execute(
                    "SELECT anvil.lock_epoch(%s)", (collection,),
                ).fetchone()
                if row != (self._epoch,):
                    raise PermissionError("Anvil writer epoch is stale or unregistered")
                yield self._read(connection, collection), _TransactionWriter(connection)


def advance_epoch(connection, investigation: str, *, expected_epoch: int) -> int:
    """Advance an existing owner, or initialize one with expected epoch zero."""
    collection = _collection(investigation)
    if type(expected_epoch) is not int or expected_epoch < 0:
        raise ValueError("invalid expected epoch")
    if expected_epoch == 0:
        row = connection.execute(
            "INSERT INTO anvil.writer VALUES (%s, 1) ON CONFLICT DO NOTHING RETURNING epoch", (collection,),
        ).fetchone()
    else:
        row = connection.execute(
            "UPDATE anvil.writer SET epoch = epoch + 1 WHERE collection = %s AND epoch = %s RETURNING epoch",
            (collection, expected_epoch),
        ).fetchone()
    if row is None:
        raise PermissionError("Anvil writer epoch changed")
    return row[0]
