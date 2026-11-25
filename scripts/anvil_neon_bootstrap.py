"""Emit owner-run bootstrap SQL or check the restricted runtime DSN without disclosing it."""
import argparse
import os

from hephaestus.state.anvil_postgres import SCHEMA, _collection

GRANTS = """
CREATE ROLE anvil_runtime LOGIN NOINHERIT NOCREATEDB NOCREATEROLE;
REVOKE ALL ON SCHEMA anvil FROM PUBLIC;
GRANT USAGE ON SCHEMA anvil TO anvil_runtime;
GRANT SELECT, INSERT ON anvil.contribution TO anvil_runtime;
GRANT EXECUTE ON FUNCTION anvil.lock_epoch(text) TO anvil_runtime;
"""


def bootstrap(investigation: str) -> str:
    return (
        "BEGIN;\n" + SCHEMA + GRANTS
        + f"INSERT INTO anvil.writer VALUES ('{_collection(investigation)}', 1);\n"
        + "COMMIT;\n"
    )


def connect_runtime():
    import psycopg
    from psycopg.conninfo import conninfo_to_dict

    dsn = os.environ["ANVIL_POSTGRES_DSN"]
    info = conninfo_to_dict(dsn)
    if info.get("user") != "anvil_runtime":
        raise PermissionError("secret role is not anvil_runtime")
    import re
    if (not re.fullmatch(r"ep-small-sound-bgci9tjx(?:\.[a-z0-9-]+)+\.neon\.tech", info.get("host", ""))
            or info.get("hostaddr") or info.get("dbname") != "neondb"):
        raise PermissionError("secret does not reference the direct Anvil Neon endpoint")
    try:
        return psycopg.connect(dsn, connect_timeout=10, sslmode="verify-full", sslrootcert="/etc/ssl/certs/ca-certificates.crt")
    except psycopg.OperationalError as error:
        diagnostic = str(error).lower()
        for fragment, reason in (
            ("password authentication failed", "password authentication rejected"),
            ("role is not permitted to log in", "runtime login disabled"),
            ("not permitted to log in", "runtime login disabled"),
            ("certificate verify", "TLS certificate verification failed"),
            ("root certificate", "TLS root certificate unavailable"),
            ("hostname", "endpoint hostname resolution or validation failed"),
            ("timeout", "endpoint connection timed out"),
            ("channel binding", "TLS channel-binding enforcement failed"),
            ("endpoint is disabled", "Neon endpoint disabled"),
            ("no password supplied", "no runtime password supplied"),
            ("password", "server rejected password or credential format"),
            ("does not exist", "server did not recognize the runtime role or database"),
            ("server closed", "endpoint closed the PostgreSQL connection"),
            ("ssl", "SSL negotiation failed"),
            ("certificate", "TLS certificate or trust-chain negotiation failed"),
            ("connection refused", "endpoint refused the connection"),
            ("invalid", "provider connection parameters were invalid"),
            ("fatal", "PostgreSQL returned a fatal startup response"),
            ("server", "PostgreSQL server startup failed"),
        ):
            if fragment in diagnostic:
                raise PermissionError(reason) from None
        raise PermissionError("verified TLS connection could not be established") from None


def verify_runtime(investigation: str) -> None:
    with connect_runtime() as connection:
        capabilities = connection.execute("""
            SELECT current_user = 'anvil_runtime',
                   has_schema_privilege(current_user, 'anvil', 'CREATE'),
                   has_table_privilege(current_user, 'anvil.contribution', 'SELECT'),
                   has_table_privilege(current_user, 'anvil.contribution', 'INSERT'),
                   has_table_privilege(current_user, 'anvil.contribution', 'UPDATE'),
                   has_table_privilege(current_user, 'anvil.contribution', 'DELETE'),
                   has_table_privilege(current_user, 'anvil.contribution', 'TRUNCATE'),
                   has_table_privilege(current_user, 'anvil.writer', 'UPDATE'),
                   has_function_privilege(current_user, 'anvil.lock_epoch(text)', 'EXECUTE'),
                   (SELECT NOT (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls)
                    FROM pg_roles WHERE rolname = current_user)
        """).fetchone()
        if capabilities != (True, False, True, True, False, False, False, False, True, True):
            raise PermissionError("runtime role privileges differ from approved grant")
        if connection.execute("SELECT anvil.lock_epoch(%s)", (_collection(investigation),)).fetchone() != (1,):
            raise PermissionError("initial investigation writer epoch is absent or changed")
    print("ANVIL_POSTGRES_RUNTIME_PRIVILEGES_VERIFIED=true")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["emit-sql", "verify-runtime"])
    parser.add_argument("investigation")
    args = parser.parse_args()
    if not args.investigation or len(args.investigation) > 128:
        parser.error("investigation identity must contain 1 to 128 characters")
    if args.action == "emit-sql":
        print(bootstrap(args.investigation))
    else:
        try:
            verify_runtime(args.investigation)
        except Exception as error:
            parser.exit(1, f"Runtime verification failed ({type(error).__name__}).\n")


if __name__ == "__main__":
    main()
