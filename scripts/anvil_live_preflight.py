"""Authenticated real cross-runner preflight; never calls provider create/delete.

The only remote writes are tiny, explicitly synthetic handoff objects under
the exact preflight activation prefix. The real investigation and approval
consumption record are only read.
"""
from datetime import timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import sleep

import anvil_global_audit_control as audit
import anvil_live_mission as mission
from anvil_neon_bootstrap import connect_runtime, verify_runtime
from hephaestus.control.anvil_attestation import AnvilAuthority
from hephaestus.control.anvil_live import publish, publish_frozen_inputs
from hephaestus.control.anvil_live_review import OffPodAuthority
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod.anvil_live_lease import AnvilPodIdentity, GLOBAL, GPU, REGION, REGIONAL
from hephaestus.providers.runpod.anvil_live_retention import AnvilRetainedReader, AnvilRetainedWriter
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing
from hephaestus.providers.runpod.execution import RunPodExecutionAdapter
from hephaestus.providers.runpod.storage import RunPodNetworkVolumeStorage
from hephaestus.state.anvil_postgres import PostgresAnvilState
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import read_verified_json


def scope():
    if (os.environ.get("GITHUB_REPOSITORY")!="kys928/Hephaestus"
            or not os.environ.get("GITHUB_REF","").startswith("refs/heads/ops/anvil-final-preflight-")
            or os.environ.get("GITHUB_RUN_ATTEMPT")!="1"):
        raise PermissionError("fresh read-only preflight branch required")
    return os.environ["GITHUB_SHA"]


def provider(root):
    config=RunPodConfig(REGION,REGIONAL,"https://s3api-eu-cz-1.runpod.io","https://api.runpod.io/v2")
    return RunPodExecutionAdapter(config,EnvironmentSecretsProvider(),state_root=root)


def main(mode):
    marker=scope()
    root=Path("/tmp")/("anvil-preflight-"+os.environ["GITHUB_RUN_ID"]+"-"+mode)
    root.mkdir(mode=0o700,exist_ok=False)
    control=provider(root/"provider")
    if AnvilPodIdentity(marker).discover(control):
        raise PermissionError("preflight must never see an allocated resource")
    if mode=="watch-prime":
        print("ANVIL_PREFLIGHT_WATCHDOG_PRIMED=true",flush=True)
        return
    if mode=="watch":
        # Prove independently hosted readiness without creating/deleting anything.
        sleep(110)
        if AnvilPodIdentity(marker).discover(control):
            raise PermissionError("unexpected provider resource")
        print("ANVIL_PREFLIGHT_WATCHDOG_NO_ALLOCATION=true",flush=True)
        return

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    artifacts=FileSystemArtifactStore(root/"artifacts")
    if sha256(mission.LOCK.read_bytes()).hexdigest()!=mission.LOCK_HASH:
        raise PermissionError("lock hash differs")
    if sha256(mission.IMAGE_QUALIFICATION.read_bytes()).hexdigest()!=mission.IMAGE_QUALIFICATION_HASH:
        raise PermissionError("effective environment receipt differs")
    archive=mission.source_archive(marker)
    source_hash=sha256(archive).hexdigest()
    from anvil_live_pod_gateway import extract_source
    extract_source(archive,root/"unpacked",source_hash)
    if sha256((root/"unpacked/requirements-anvil-live.lock").read_bytes()).hexdigest()!=mission.LOCK_HASH:
        raise PermissionError("gateway-extracted lock differs")
    bundle=json.loads(mission.BUNDLE.read_bytes())
    plan_ref=publish_frozen_inputs(bundle,artifacts)
    plan=bundle["plan"]["document"]
    verify_runtime(plan["investigation"])
    board=PostgresAnvilState(connect_runtime,plan["investigation"],1,lambda _:False,lambda *_:False)
    initial=board.replay(plan["investigation"],expected_frontier=bundle["expected_initial_frontier"])
    if initial.contributions:
        raise PermissionError("actual research investigation is not empty")
    cfg=RunPodConfig("EU-CZ-1",REGIONAL,"https://s3api-eu-cz-1.runpod.io","https://api.runpod.io/v2")
    store=RunPodNetworkVolumeStorage.from_boto3(cfg,EnvironmentSecretsProvider())
    store.head_volume()
    mailbox=AnvilRetainedWriter(store.client,REGIONAL,marker)
    reader=AnvilRetainedReader(store.client,REGIONAL,marker)
    if mailbox.allocation_intent(plan_ref) is not None:
        raise PermissionError("actual one-shot approval is already consumed")
    audit.WATCHDOG_FILE="anvil-live-preflight-watchdog.yml"
    audit.LEASE_MINUTES=5
    deadline=audit.active_watchdog(marker)
    stop=min(deadline,mission.now()+timedelta(seconds=90))
    binding={"schema":"anvil.cross-runner-preflight.v1","activation":marker,
        "plan_ref":plan_ref,"source_sha256":source_hash,"frontier":initial.frontier,"fixture":True}
    if mode=="reviewer":
        authority=OffPodAuthority(Ed25519PrivateKey.generate())
        mailbox.publish_message("ready",dict(binding,public_key_hex=authority.verifier.public_key_hex,
            public_key_ref=authority.verifier.public_key_ref))
        request,_=mission.wait_message(mailbox,"slot-0-grant-request",stop)
        if read_verified_json(reader,request["ref"])!=binding:
            raise PermissionError("independent fixture readback differs")
        signed=authority.sign(artifacts,dict(binding,receipt_ref=request["ref"],independent_readback=True))
        mailbox.retain(signed,artifacts)
        mailbox.publish_message("slot-0-grant",{"ref":signed})
        print("ANVIL_PREFLIGHT_INDEPENDENT_REVIEW=true",flush=True)
    elif mode=="controller":
        ready,_=mission.wait_message(mailbox,"ready",stop)
        mission.check_reviewer_active(marker)
        if any(ready.get(k)!=v for k,v in binding.items()):
            raise PermissionError("independent preflight source/frontier differs")
        authority=AnvilAuthority(ready["public_key_hex"],ready["public_key_ref"])
        ref=publish(artifacts,binding)
        mailbox.retain(ref,artifacts)
        mailbox.publish_message("slot-0-grant-request",{"ref":ref})
        reviewed,_=mission.wait_message(mailbox,"slot-0-grant",stop)
        mission.copy_ref(reader,artifacts,reviewed["ref"])
        authority.verify(artifacts,reviewed["ref"],schema=binding["schema"],
            expected=dict(binding,receipt_ref=ref,independent_readback=True))
        pricing=ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"])
        gpu=pricing.quote_gpu(gpu_id=GPU,datacenter_id=REGION)
        stock=control.select_gpu([GPU])
        status,volume=control._request("GET","global-volumes/"+GLOBAL)
        account=pricing.account()
        total=gpu.price_usd_per_hour+mission.DISK_ALLOWANCE
        if (stock!=GPU or gpu.gpu_memory_gb<48 or status!=200 or volume.get("id")!=GLOBAL
                or total>mission.MAX_RATE or account.autopay_enabled
                or account.credit_balance_usd<=total+account.already_spending_usd_per_hour):
            raise PermissionError("actual placement/prepaid gate failed")
        print("ANVIL_FINAL_PREFLIGHT="+json.dumps({"source_sha256":source_hash,"plan_ref":plan_ref,
            "initial_frontier":initial.frontier,"gpu":GPU,"region":REGION,"gpu_hour_usd":str(gpu.price_usd_per_hour),
            "conservative_disk_hour_usd":str(mission.DISK_ALLOWANCE),"all_in_ceiling_usd":str(total),
            "independent_watchdog_handshake":True,"separate_signed_s3_readback":True,
            "real_approval_unconsumed":True,"allocation_performed":False},sort_keys=True),flush=True)
    else:
        raise ValueError("unknown preflight role")


if __name__=="__main__":
    try:
        main(sys.argv[1])
    except Exception as error:
        sys.exit("Anvil no-allocation preflight stopped ("+type(error).__name__+")")
