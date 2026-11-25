"""Independent GitHub-runner watchdog for one exact Anvil A40 activation."""
from datetime import datetime, timedelta, timezone
import os
import json
from urllib.request import Request, urlopen
from pathlib import Path
import sys

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod.anvil_live_lease import AnvilPodIdentity, REGION, REGIONAL, MAX_SECONDS
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.execution import RunPodExecutionAdapter


def main():
    if (os.environ.get("GITHUB_REPOSITORY") != "kys928/Hephaestus"
            or os.environ.get("GITHUB_REF") != "refs/heads/feature/anvil-integration"
            or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"):
        raise PermissionError("exact one-shot Anvil integration execution required")
    identity = AnvilPodIdentity(os.environ["GITHUB_SHA"])
    config = RunPodConfig(REGION, REGIONAL, "https://s3api-eu-cz-1.runpod.io",
                          "https://api.runpod.io/v2")
    root = Path("/tmp") / ("anvil-live-watchdog-" + os.environ["GITHUB_RUN_ID"])
    provider = RunPodExecutionAdapter(config, EnvironmentSecretsProvider(), state_root=root)
    mode = sys.argv[1]
    if mode == "prime":
        if identity.discover(provider):
            raise PermissionError("existing mission Pod; do not arm another allocation")
        print("ANVIL_GPU_WATCHDOG_READY=true", flush=True)
    elif mode == "watch":
        url = ("https://api.github.com/repos/kys928/Hephaestus/actions/runs/"
               + os.environ["GITHUB_RUN_ID"] + "/jobs")
        request = Request(url, headers={"Authorization":"Bearer "+os.environ["GITHUB_TOKEN"]})
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
        jobs = [row for row in payload["jobs"] if row["name"] == "watchdog" and row["status"] == "in_progress"]
        if len(jobs) != 1:
            raise PermissionError("watchdog job identity is not unique and active")
        started = datetime.fromisoformat(jobs[0]["started_at"].replace("Z", "+00:00"))
        deadline = started + timedelta(seconds=MAX_SECONDS)
        if not timedelta(0) <= datetime.now(timezone.utc) - started <= timedelta(minutes=5):
            raise PermissionError("watchdog start is stale or unbound")
        result = identity.monitor(provider, deadline=deadline)
        print("ANVIL_GPU_WATCHDOG_TERMINAL=" + result, flush=True)
    else:
        raise ValueError("expected prime or watch")


if __name__ == "__main__":
    main()
