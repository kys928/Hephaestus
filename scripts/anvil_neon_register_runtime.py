"""Operator-only handoff. Prompt for the Neon owner DSN and register the runtime DSN."""
import argparse
from contextlib import contextmanager
from pathlib import Path
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


@contextmanager
def stage(name):
    try:
        yield
    except RegistrationFailure:
        raise
    except Exception as error:
        state = getattr(error, "sqlstate", None)
        code = state if isinstance(state, str) and re.fullmatch(r"[A-Z0-9]{5}", state) else "unavailable"
        kind = type(error).__name__
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", kind):
            kind = "Error"
        raise RegistrationFailure(f"stage={name} error={kind} sqlstate={code}") from None


class RegistrationFailure(RuntimeError):
    """A report containing only fixed stage names and sanitized error metadata."""


def tls_root(ca_bundle):
    if ca_bundle is None and sys.platform == "win32":
        import certifi
        ca_bundle = certifi.where()
    if ca_bundle is None:
        return "system"
    return str(Path(ca_bundle).resolve(strict=True))


def checked_command(args, *, stdin=None):
    result = subprocess.run(args, input=stdin, text=True, capture_output=True, timeout=25)
    if result.returncode:
        raise PermissionError("GitHub CLI authentication or repository-secret operation failed")
    return result.stdout


def register(*, replace_existing=False, ca_bundle=None, reuse_runtime=False):
    with stage("dependencies"):
        import psycopg
        from psycopg import sql
        from psycopg.conninfo import conninfo_to_dict, make_conninfo

    with stage("github_preflight"):
        gh = shutil.which("gh")
        if gh is None or not sys.stdin.isatty():
            raise RuntimeError("requires private interactive terminal with GitHub CLI")
        checked_command([gh, "auth", "status", "--hostname", "github.com"])
        existing = checked_command([gh, "secret", "list", "--repo", REPOSITORY, "--json", "name"])
        if not replace_existing and any(row.get("name") == SECRET_NAME for row in json.loads(existing)):
            raise FileExistsError("runtime secret already exists")

    with stage("tls_configuration"):
        root = tls_root(ca_bundle)
    user = "anvil_runtime" if reuse_runtime else "neondb_owner"
    with stage("runtime_dsn_input" if reuse_runtime else "owner_dsn_input"):
        source = conninfo_to_dict(getpass(f"Paste the direct Neon DSN for {user}: "))
        host = source.get("host", "")
        if not re.fullmatch(re.escape(NEON_ENDPOINT) + r"(?:\.[a-z0-9-]+)+\.neon\.tech", host):
            raise PermissionError("wrong direct Neon endpoint")
        if source.get("user") != user or source.get("dbname") != "neondb" or source.get("hostaddr"):
            raise PermissionError("wrong role or database")
        if not source.get("password"):
            raise ValueError("DSN does not include a password")
        common = dict(host=host, port="5432", dbname="neondb", sslmode="verify-full", connect_timeout=10)
        password = source["password"] if reuse_runtime else token_urlsafe(36)
        runtime_dsn = make_conninfo(**common, user="anvil_runtime", password=password)

    if not reuse_runtime:
        with stage("owner_connect"):
            owner_dsn = make_conninfo(**common, user=user, password=source["password"])
            owner = psycopg.connect(owner_dsn, sslrootcert=root)
        try:
            with stage("role_profile"):
                profile = owner.execute("""
                    SELECT NOT (r.rolsuper OR r.rolcreatedb OR r.rolcreaterole OR r.rolreplication OR r.rolbypassrls),
                           NOT pg_has_role(r.oid, elevated.oid, 'MEMBER'::text)
                    FROM pg_catalog.pg_roles AS r
                    JOIN pg_catalog.pg_roles AS elevated ON elevated.rolname = 'neon_superuser'
                    WHERE r.rolname = 'anvil_runtime'
                """).fetchone()
                if profile != (True, True):
                    raise PermissionError("runtime role missing or elevated")
            with stage("alter_runtime_role"):
                owner.execute(sql.SQL("ALTER ROLE anvil_runtime LOGIN PASSWORD {}").format(sql.Literal(password)))
            with stage("commit_runtime_role"):
                owner.commit()
        finally:
            owner.close()

    with stage("runtime_connect"):
        runtime = psycopg.connect(runtime_dsn, sslrootcert=root)
    with runtime, stage("verify_runtime_role"):
        if runtime.execute("SELECT current_user").fetchone() != ("anvil_runtime",):
            raise PermissionError("runtime credential authenticated as an unexpected role")

    with stage("publish_github_secret"):
        # Publish a portable DSN. The consumer selects its own verified CA bundle.
        checked_command([gh, "secret", "set", SECRET_NAME, "--repo", REPOSITORY], stdin=runtime_dsn)
    print("ANVIL_POSTGRES_DSN registered; restricted login verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--reuse-runtime", action="store_true", help="verify and publish an existing runtime DSN without role changes")
    args = parser.parse_args()
    try:
        register(replace_existing=args.replace_existing, ca_bundle=args.ca_bundle, reuse_runtime=args.reuse_runtime)
    except RegistrationFailure as error:
        sys.exit(f"Runtime registration stopped: {error}. No credentials were printed.")
