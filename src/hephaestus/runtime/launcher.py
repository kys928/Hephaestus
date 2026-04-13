from __future__ import annotations

import subprocess
from dataclasses import dataclass, field


@dataclass(slots=True)
class LaunchResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    cwd: str | None = None
    env_keys: list[str] = field(default_factory=list)


def launch_subprocess(
    command: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> LaunchResult:
    completed = subprocess.run(command, capture_output=True, text=True, check=False, cwd=cwd, env=env or None)
    return LaunchResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        cwd=cwd,
        env_keys=sorted((env or {}).keys()),
    )
