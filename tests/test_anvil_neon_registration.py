"""The credential helper must report only safe diagnostic fields."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/anvil_neon_register_runtime.py"
spec = importlib.util.spec_from_file_location("anvil_neon_register_runtime", path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.mark.parametrize(("sqlstate", "expected"), [("42501", "42501"), ("private-value", "unavailable")])
def test_registration_redacts_driver_error(sqlstate, expected):
    class ProgrammingError(Exception):
        pass

    error = ProgrammingError("postgresql://owner:secret@host/database")
    error.sqlstate = sqlstate
    with pytest.raises(helper.RegistrationFailure) as rejected:
        with helper.stage("alter_runtime_role"):
            raise error
    assert str(rejected.value) == f"stage=alter_runtime_role error=ProgrammingError sqlstate={expected}"
    assert rejected.value.__suppress_context__


def test_ca_bundle_override(tmp_path):
    bundle = tmp_path / "certificates.pem"
    bundle.touch()
    assert helper.tls_root(bundle) == str(bundle.resolve())
