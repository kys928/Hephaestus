"""Operator-only handoff. Prompt for the Neon owner DSN and register the runtime DSN."""
from getpass import getpass
import json
import re
from secrets import token_urlsafe
import shutil
import subprocess
import sys

REPOSITORY = "kys928/Hephaestus"
SECRET_NAME = "ANVIL_POSTGRES_DSN"
NEON_ENDPOINT = "ep-small-sound-bgci9tjx"


def checked_command(args, *, stdin=None):
    result = subprocess.run(args, input=stdin, text=True, capture_output=True, timeout=25)
    if result.returncode:
        raise PermissionError("GitHub CLI authentication or repository-secret operation failed")
    return result.stdout


def register():
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    gh = shutil.which("gh")
    if gh is None or not sys.stdin.isatty():
        raise RuntimeError("use a private interactive terminal with the GitHub CLI installed")
    checked_command([gh, "auth", "status", "--hostname", "github.com"])
    existing = checked_command([gh, "secret", "list", "--repo", REPOSITORY, "--json", "name"])
    if any(row.get("name") == SECRET_NAME for row in json.loads(existing)):
        raise FileExistsError("runtime secret already exists; verify it instead of replacing it")
    owner = conninfo_to_dict(getpass("Paste the direct Neon owner DSN for hephaestus-anvil: "))
    host = owner.get("host", "")
    if not re.fullmatch(re.escape(NEON_ENDPOINT) + r"(?:\.[a-z0-9-]+)+\.neon\.tech", host):
        raise PermissionError("direct endpoint identity does not match hephaestus-anvil")
    if owner.get("user") != "neondb_owner" or owner.get("dbname") != "neondb" or owner.get("hostaddr"):
        raise PermissionError("expected the dedicated project's direct owner connection")
    owner_dsn = make_conninfo(host=host, port="5432", user="neondb_owner", dbname="neondb",
                              password=owner["password"], sslmode="verify-full", sslrootcert="system",
                              connect_timeout=10)
    runtime_password = token_urlsafe(36)
    runtime_dsn = make_conninfo(host=host, port="5432", user="anvil_runtime", dbname="neondb",
                                password=runtime_password, sslmode="verify-full", sslrootcert="system",
                                connect_timeout=10)
    with psycopg.connect(owner_dsn) as connection:
        profile = connection.execute("""
            SELECT NOT (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls),
                   NOT pg_has_role(rolname, 'neon_superuser', 'MEMBER')
            FROM pg_roles WHERE rolname = 'anvil_runtime'
        """).fetchone()
        if profile != (True, True):
            raise PermissionError("runtime role is absent or retains elevated privileges")
        # Neon requires plaintext on the authenticated TLS connection. SQL
        # construction is local and the credential never reaches the terminal.
        connection.execute(sql.SQL("ALTER ROLE anvil_runtime LOGIN PASSWORD {}").format(
            sql.Literal(runtime_password)
        ))
        checked_command([gh, "secret", "set", SECRET_NAME, "--repo", REPOSITORY], stdin=runtime_dsn)

    with psycopg.connect(runtime_dsn) as connection:
        if connection.execute("SELECT current_user").fetchone() != ("anvil_runtime",):
            raise PermissionError("runtime credential was not accepted as its restricted role")
    print("ANVIL_POSTGRES_DSN registered; restricted login verified.")


if __name__ == "__main__":
    try:
        register()
    except Exception as error:
        sys.exit(f"Runtime registration stopped ({type(error).__name__}). No credentials were printed.")
