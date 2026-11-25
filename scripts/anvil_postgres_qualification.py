"""Synthetic PostgreSQL writer fencing gate. Requires an empty disposable database."""
import os
import subprocess
import sys

import psycopg


def main():
    dsn = os.environ["ANVIL_QUALIFICATION_DSN"]
    with psycopg.connect(dsn, autocommit=True) as owner, psycopg.connect(dsn) as producer:
        owner.execute("""
            CREATE SCHEMA anvil_qualification;
            CREATE TABLE anvil_qualification.owner (
                id integer PRIMARY KEY CHECK (id = 1), epoch bigint NOT NULL
            );
            INSERT INTO anvil_qualification.owner VALUES (1, 1);
            CREATE TABLE anvil_qualification.accepted (
                id text PRIMARY KEY, body text NOT NULL
            );
            CREATE FUNCTION anvil_qualification.admit(claim bigint, key text, payload text)
            RETURNS boolean LANGUAGE plpgsql AS $$
            DECLARE current_epoch bigint;
            BEGIN
                SELECT epoch INTO current_epoch FROM anvil_qualification.owner
                    WHERE id = 1 FOR UPDATE;
                IF claim <> current_epoch THEN RAISE EXCEPTION 'stale owner'; END IF;
                INSERT INTO anvil_qualification.accepted VALUES (key, payload)
                    ON CONFLICT (id) DO NOTHING;
                IF (SELECT body FROM anvil_qualification.accepted WHERE id = key) <> payload
                    THEN RAISE EXCEPTION 'conflicting retry';
                END IF;
                RETURN true;
            END $$;
        """)
        admit = "SELECT anvil_qualification.admit(%s, %s, %s)"
        producer.execute(admit, (1, "first", "original"))
        owner.execute("SET lock_timeout = '250ms'")
        try:
            owner.execute("UPDATE anvil_qualification.owner SET epoch = 2 WHERE id = 1")
        except psycopg.errors.LockNotAvailable:
            pass
        else:
            raise AssertionError("replacement overlapped an outstanding admission")
        producer.commit()
        owner.execute("UPDATE anvil_qualification.owner SET epoch = 2 WHERE id = 1")
        for arguments, message in [((1, "stale", "bad"), "stale owner"),
                                   ((2, "first", "changed"), "conflicting retry")]:
            try:
                owner.execute(admit, arguments)
            except psycopg.errors.RaiseException as error:
                assert message in str(error)
            else:
                raise AssertionError(message)
        owner.execute(admit, (2, "first", "original"))

        # Drop a different process after its write but before commit.
        command = (
            "import os, psycopg; "
            "c=psycopg.connect(os.environ['ANVIL_QUALIFICATION_DSN']); "
            "c.execute(\"SELECT anvil_qualification.admit(2, 'interrupted', 'bad')\"); os._exit(23)"
        )
        assert subprocess.run([sys.executable, "-c", command]).returncode == 23
        owner.execute(admit, (2, "second", "replacement"))

    with psycopg.connect(dsn) as reader:
        assert reader.execute("SELECT * FROM anvil_qualification.accepted ORDER BY id").fetchall() == [
            ("first", "original"), ("second", "replacement"),
        ]
    print("PostgreSQL synthetic epoch fencing, retry, interrupted transaction and independent readback passed")


if __name__ == "__main__":
    main()
