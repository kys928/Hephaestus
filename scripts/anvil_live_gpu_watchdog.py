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
from hephaestus.providers.runpod.storage import RunPodNetworkVolumeStorage
from hephaestus.providers.runpod.anvil_live_retention import AnvilRetainedWriter
from hephaestus.providers.runpod.anvil_hardware import GPU_NAMES, valid_region


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
    s3config=RunPodConfig("EU-CZ-1",REGIONAL,config.s3_endpoint_url,config.api_base_url)
    storage=RunPodNetworkVolumeStorage.from_boto3(s3config,EnvironmentSecretsProvider())
    storage.head_volume()
    mailbox=AnvilRetainedWriter(storage.client,REGIONAL,identity.marker)
    plan=json.loads((Path(__file__).resolve().parents[1]/"configs/research/anvil_diagnosis_planner_first_live_v1.json").read_bytes())["plan"]["ref"]
    if mode == "prime":
        if mailbox.allocation_intent(plan) is not None:
            raise PermissionError("one-use allocation approval already consumed")
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
        # Independently inspect the one-use intent before accepting a dynamic Pod.
        def placement():
            intent=mailbox.allocation_intent(plan)
            if (not intent or intent.get("activation")!=identity.marker or intent.get("operation_id")!=identity.operation_id
                    or intent.get("gpu") not in GPU_NAMES or not valid_region(intent.get("region"))
                    or intent.get("maximum_usd")!="0.65" or intent.get("maximum_pods")!=1):
                raise PermissionError("watchdog cannot corroborate selected allocation intent")
            return {"region":intent["region"],"sku":intent["gpu"]}
        # A mailbox failure during active supervision fails closed through
        # monitor's finally-cleanup path, not an unguarded worker.
        result = identity.monitor(provider, deadline=deadline,placement=placement)
        print("ANVIL_GPU_WATCHDOG_TERMINAL=" + result, flush=True)
    else:
        raise ValueError("expected prime or watch")


if __name__ == "__main__":
    main()
