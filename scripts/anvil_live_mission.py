"""Exact bounded first-live operator and independently hosted receipt reviewer.

Deployment is intentionally inactive until the reviewed workflow templates are
explicitly activated for this single approved experiment. The GPU gets neither
the database DSN, provider keys, S3 keys, nor the signing private key.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from secrets import token_hex
import sqlite3
import subprocess
import sys
from time import sleep
from urllib.error import URLError

import anvil_global_audit_control as audit
from anvil_neon_bootstrap import connect_runtime, verify_runtime
from hephaestus.control.anvil_attestation import AnvilAuthority
from hephaestus.control.anvil_context import AnvilAdmission, material_collection
from hephaestus.control.anvil_dispatch import AnvilDispatch
from hephaestus.control.anvil_live import AnvilLiveController, publish, publish_frozen_inputs
from hephaestus.control.anvil_live_review import OffPodAuthority, issue_slot, attest_retained_slot, build_review_packet
from hephaestus.control.anvil_live_runtime import verify_loaded_runtime
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.jobs import SQLiteJobQueue
from hephaestus.production.original_cuda_identity import verified_original_roles
from hephaestus.providers.runpod.anvil_live_lease import (
    AnvilPodIdentity, BASE_NAME, REGION, REGIONAL, GLOBAL, GPU, IMAGE,
    MAX_RATE, DISK_ALLOWANCE, CLEANUP_RESERVE_SECONDS,
)
from hephaestus.providers.runpod.anvil_live_retention import AnvilRetainedReader, AnvilRetainedWriter
from hephaestus.providers.runpod.anvil_live_transport import AnvilGatewayClient
from hephaestus.providers.runpod.anvil_live_startup import (
    gateway_launch, health_probe, provider_state, startup_log_signals, WorkerHealthFailure,
)
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing
from hephaestus.providers.runpod.anvil_dynamic_placement import discover
from hephaestus.providers.runpod.anvil_hardware import GPU_NAMES, valid_region
from hephaestus.providers.runpod.anvil_placement_diagnostics import PlacementFailure, inspect_placement, safe_failure
from hephaestus.providers.runpod.execution import RunPodExecutionAdapter
from hephaestus.providers.runpod.global_volume import GlobalVolumeConfig
from hephaestus.providers.runpod.storage import RunPodNetworkVolumeStorage
from hephaestus.schemas.anvil import ContributionOrigin
from hephaestus.state.anvil import AnvilState
from hephaestus.state.anvil_postgres import PostgresAnvilState
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes, read_verified_json

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-anvil-live.lock"
IMAGE_QUALIFICATION = ROOT / "configs/research/anvil_runtime_image_qualification.json"
BUNDLE = ROOT / "configs/research/anvil_diagnosis_planner_first_live_v1.json"
LOCK_HASH = "01dd1ff5af4241ac9b6f97f02009dcf21f30836fb3fa1687576eeece96d93c92"
IMAGE_QUALIFICATION_HASH = "58250b8f506bc80f1e53c03a0cb88f3674155a59e4f9418f09a2abf73d6c5ffd"


def now():
    return datetime.now(timezone.utc)


def source_archive(marker):
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=marker:
        raise PermissionError("checked out source differs from activation")
    tar = subprocess.check_output(["git","archive","--format=tar",marker,"src","scripts","configs/models",
        "docs/evidence/role-model-stack-v1/certification.json","requirements-anvil-live.lock"],cwd=ROOT)
    packed = gzip.compress(tar,mtime=0)
    if len(tar)>64*1024**2 or len(packed)>16*1024**2:
        raise ValueError("source archive exceeds gateway limits")
    return packed


def independent_deadline(marker):
    # Reuse the independently hosted, previously exercised Actions handshake.
    audit.WATCHDOG_FILE="anvil-live-gpu-watchdog.yml"
    audit.LEASE_MINUTES=60
    return audit.active_watchdog(marker)


def check_reviewer_active(marker):
    url = audit.GITHUB + "/actions/runs/" + os.environ["GITHUB_RUN_ID"] + "/jobs"
    jobs = audit.read_json(url,os.environ["GITHUB_TOKEN"])["jobs"]
    reviewers = [j for j in jobs if j["name"]=="reviewer" and j["status"]=="in_progress"
                 and j.get("head_sha", marker)==marker]
    if len(reviewers)!=1 or not any(s["name"]=="Independently review live evidence" and
            s["status"]=="in_progress" for s in reviewers[0]["steps"]):
        raise PermissionError("separate reviewer is not running")


def wait_message(mailbox,name,deadline,*,terminal=False):
    while now()<deadline:
        if terminal:
            try:
                return None, mailbox.read_message("terminal-request")
            except FileNotFoundError:
                pass
        try:
            doc = mailbox.read_message(name)
            return doc, None
        except FileNotFoundError:
            sleep(2)
    raise TimeoutError("bounded independent handoff expired")


def copy_ref(reader, artifacts, ref):
    doc=read_verified_json(reader,ref,limit=2*1024**2)
    publish(artifacts,doc,ref)
    return doc


class Mission:
    def __init__(self,mode):
        self.marker=os.environ["GITHUB_SHA"]
        if (os.environ.get("GITHUB_REPOSITORY")!="kys928/Hephaestus"
                or os.environ.get("GITHUB_REF")!="refs/heads/feature/anvil-integration"
                or os.environ.get("GITHUB_RUN_ATTEMPT")!="1"
                or os.environ.get("ANVIL_FIRST_LIVE_ACTIVATION")!="approved-dynamic-one-gpu-0.65-usd-3600s"):
            raise PermissionError("fresh exact approved one-shot activation required")
        self.identity=AnvilPodIdentity(self.marker)
        self.selected=None
        self.root=Path("/tmp")/("anvil-live-"+os.environ["GITHUB_RUN_ID"]+"-"+mode)
        self.root.mkdir(mode=0o700,exist_ok=False)
        self.artifacts=FileSystemArtifactStore(self.root/"artifacts")
        self.repository=SQLiteStateRepository(self.root/"state.sqlite")
        self.queue=SQLiteJobQueue(self.root/"jobs.sqlite",maximum_attempts=1)
        self.bundle=json.loads(BUNDLE.read_bytes())
        self.plan_ref=publish_frozen_inputs(self.bundle,self.artifacts)
        self.plan=self.bundle["plan"]["document"]
        self.material=frozenset(self.plan["material_refs"])
        self.source=source_archive(self.marker)
        self.source_hash=sha256(self.source).hexdigest()
        image_raw=IMAGE_QUALIFICATION.read_bytes()
        if sha256(image_raw).hexdigest()!=IMAGE_QUALIFICATION_HASH:
            raise PermissionError("independently qualified environment receipt changed")
        self.image_qualification=json.loads(image_raw)
        if sha256(LOCK.read_bytes()).hexdigest()!=LOCK_HASH or self.image_qualification.get("lock_sha256")!=LOCK_HASH:
            raise PermissionError("exact effective image qualification missing")
        self.config=RunPodConfig(REGION,REGIONAL,"https://s3api-eu-cz-1.runpod.io","https://api.runpod.io/v2")
        secrets=EnvironmentSecretsProvider()
        self.provider=RunPodExecutionAdapter(self.config,secrets,state_root=self.root/"provider")
        s3config=RunPodConfig("EU-CZ-1",REGIONAL,self.config.s3_endpoint_url,self.config.api_base_url)
        storage=RunPodNetworkVolumeStorage.from_boto3(s3config,secrets)
        storage.head_volume()
        self.mailbox=AnvilRetainedWriter(storage.client,REGIONAL,self.marker)
        # Distinct remote reads; separate reviewer process receives its own client.
        self.reader=AnvilRetainedReader(storage.client,REGIONAL,self.marker)
        verify_runtime(self.plan["investigation"])
        self.dispatch=self.make_dispatch(frozenset())
        board=self.dispatch.admission._board
        board.replay(self.plan["investigation"],expected_frontier=self.bundle["expected_initial_frontier"])
        stack,_=verified_original_roles(self.plan["registry_ref"])
        spec=stack.resolve("diagnosis")
        origin=ContributionOrigin(**{k:self.plan[k] for k in ("investigation","run_id","lineage_id")},
            role="diagnosis",inference_ref=self.plan_ref,context_frontier=self.bundle["expected_initial_frontier"],
            idempotency_key="frozen-material",model_id=spec.model_id,model_revision=spec.revision,
            adapter_ref="sha256:"+spec.adapter.sha256)
        for ref in self.material:
            self.dispatch.admission.admit_material(origin,ref)

    def make_dispatch(self,trusted):
        board=PostgresAnvilState(connect_runtime,self.plan["investigation"],1,
            lambda origin: origin.inference_ref in trusted,
            lambda origin,ref: any(row.get("ref")==ref for row in self.repository.all(material_collection(origin))))
        return AnvilDispatch(AnvilAdmission(board,self.artifacts,self.plan["registry_ref"],
            self.material,trusted,self.repository),self.queue,self.plan_ref)

    def retain(self,*refs):
        for ref in refs:
            self.mailbox.retain(ref,self.artifacts)

    def import_refs(self,*refs):
        return [copy_ref(self.reader,self.artifacts,ref) for ref in refs]

    def runtime_args(self,pod_id):
        return dict(pod_id=pod_id,source_sha256=self.source_hash,
                    lock_sha256=LOCK_HASH,image_qualification=self.image_qualification,
                    gpu_sku=self.selected["gpu"])

    def provider_check(self,pod_id):
        pod=self.provider.get_pod(pod_id)
        if not self.selected or not self.identity.qualified(pod,region=self.selected["region"],sku=self.selected["gpu"]):
            raise PermissionError("provider worker differs from approved scope")
        return pod

    def checkpoint(self):
        ref=self.dispatch.admission._board.checkpoint(self.plan["investigation"],self.artifacts)
        self.retain(ref)
        return ref

    def backup_local(self):
        # Export complete SQLite snapshots, not a database copied without its WAL.
        out=self.root/"backup"
        out.mkdir(exist_ok=True)
        for name in ("state.sqlite","jobs.sqlite"):
            with sqlite3.connect(self.root/name) as source, sqlite3.connect(out/name) as target:
                source.backup(target)
                if target.execute("PRAGMA quick_check").fetchone()!=("ok",):
                    raise RuntimeError("local control backup integrity failure")


def quote(mission):
    pricing=ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"])
    def report(doc):
        ref=publish(mission.artifacts,doc)
        mission.retain(ref)
        print("ANVIL_PLACEMENT_DIAGNOSTIC="+json.dumps(dict(doc,diagnostic_ref=ref),sort_keys=True),flush=True)
    for candidate in discover(pricing,report):
        config=RunPodConfig(candidate["region"],REGIONAL,mission.config.s3_endpoint_url,mission.config.api_base_url)
        provider=RunPodExecutionAdapter(config,EnvironmentSecretsProvider(),state_root=mission.root/"provider")
        try:
            doc=inspect_placement(pricing,provider,region=candidate["region"],gpu=candidate["gpu"],report=report)
        except PlacementFailure:
            continue  # read-only fallback before one-use allocation intent
        mission.selected=doc
        mission.provider=provider
        ref=publish(mission.artifacts,doc)
        mission.retain(ref)
        print("ANVIL_DYNAMIC_SELECTED="+json.dumps(doc,sort_keys=True),flush=True)
        return ref
    raise PermissionError("no eligible funded Global GPU placement")


def allocate(mission,authority,deadline,price_ref,known):
    if mission.identity.discover(mission.provider) or now()+timedelta(minutes=15)>=deadline:
        raise PermissionError("existing Pod or insufficient remaining bounded lease")
    if not mission.selected:
        raise PermissionError("no selected qualified placement")
    region,sku=mission.selected["region"],mission.selected["gpu"]
    if mission.provider.select_gpu([sku])!=sku:
        raise PermissionError("selected stock disappeared before intent")
    token=token_hex(32)
    print("::add-mask::"+token,flush=True)
    source=(ROOT/"scripts/anvil_live_pod_gateway.py").read_bytes()
    entrypoint,cmd=gateway_launch(source)
    body={"name":BASE_NAME,"gpu":{"id":sku,"count":1},"cloud":"SECURE","dataCenterIds":[region],
          "image":IMAGE,"disk":20,"mounts":GlobalVolumeConfig(GLOBAL).mounts(),"ports":["8787/http"],
          "entrypoint":entrypoint,"cmd":cmd,
          "env":{"ANVIL_LIVE_ACTIVATION":mission.marker,"ANVIL_LIVE_TOKEN":token,
                 "ANVIL_PRICE_REF":price_ref,"ANVIL_GPU_SKU":sku,
                 "ANVIL_DEADLINE_UTC":deadline.isoformat(),"ANVIL_SOURCE_SHA256":mission.source_hash,
                 "ANVIL_PACKAGE_LOCK_SHA256":LOCK_HASH,"ANVIL_AUTHORITY_PUBLIC_HEX":authority.public_key_hex,
                 "ANVIL_AUTHORITY_REF":authority.public_key_ref}}
    mission.mailbox.consume_allocation(mission.plan_ref,{
        "schema":"anvil.first-live-allocation-intent.v1","plan_ref":mission.plan_ref,
        "activation":mission.marker,"source_sha256":mission.source_hash,"price_ref":price_ref,
        "operation_id":mission.identity.operation_id,"deadline":deadline.isoformat(),
        "gpu":sku,"region":region,"maximum_usd":"0.65","maximum_pods":1})
    # One durable provider intent. A lost acknowledgement is reconciled by the
    # independent marker watcher; never invoke this allocator twice.
    pod=mission.provider._create_pod(body,operation_id=mission.marker,gpu_type_ids=[sku])
    # The create response need not contain the full provider representation.
    # Record the exact ID before checking authenticated full readback.
    pod_id=pod["id"]
    if not re.fullmatch(r"[A-Za-z0-9]{4,40}",pod_id):
        raise PermissionError("invalid created provider Pod identity")
    known.add(pod_id)
    print("ANVIL_LIVE_EXACT_POD="+pod_id,flush=True)
    mission.provider_check(pod_id)
    return pod_id,AnvilGatewayClient(pod_id,token,deadline)


def await_health(client,deadline,*,read_provider=None):
    observations=[]
    attempts=0
    state={"status":"unobserved"}
    next_provider=now()
    while now()<deadline:
        health=health_probe(client)
        attempts+=1
        if health["reason"]!="ready" and read_provider is not None and now()>=next_provider:
            if (deadline-now()).total_seconds()>35:
                try:
                    state=provider_state(read_provider())
                except PermissionError:
                    raise
                except Exception as exc:
                    state=safe_failure("provider_startup_read",exc)
                next_provider=now()+timedelta(seconds=15)
        row={"health":health,"provider":state}
        if not observations or row!={k:v for k,v in observations[-1].items() if k!="observed_at"}:
            observed=dict(row,observed_at=now().isoformat())
            if len(observations)>=32:
                observations.pop(1)  # keep initial observation and recent transitions
            observations.append(observed)
            print("ANVIL_WORKER_HEALTH_OBSERVATION="+json.dumps(observed,sort_keys=True),flush=True)
        if health["reason"]=="ready":
            return observations
        if state.get("status") in {"ERROR","EXITED","TERMINATED","STOPPED"} or state.get("runtimeStatus") in {"ERROR","EXITED","TERMINATED","STOPPED"}:
            raise WorkerHealthFailure(observations,attempts,"provider_terminal_before_health")
        sleep(min(3,max(0,(deadline-now()).total_seconds())))
    raise WorkerHealthFailure(observations,attempts)


def controller(mission):
    hard_deadline=independent_deadline(mission.marker)
    deadline=hard_deadline-timedelta(seconds=CLEANUP_RESERVE_SECONDS+60)
    ready,_=wait_message(mission.mailbox,"ready",min(deadline,now()+timedelta(minutes=3)))
    check_reviewer_active(mission.marker)
    if (ready.get("activation")!=mission.marker or ready.get("run_id")!=os.environ["GITHUB_RUN_ID"]
            or ready.get("plan_ref")!=mission.plan_ref or ready.get("deadline")!=deadline.isoformat()):
        raise PermissionError("independent review readiness differs")
    authority=AnvilAuthority(ready["public_key_hex"],ready["public_key_ref"])
    if "sha256:"+sha256(bytes.fromhex(authority.public_key_hex)).hexdigest()!=authority.public_key_ref:
        raise PermissionError("independent authority key binding differs")
    ctl=AnvilLiveController(mission.dispatch,mission.repository,authority,mission.make_dispatch)
    for ref in (mission.plan_ref,*mission.plan["material_refs"]):
        mission.retain(ref)
    slots=[]
    known=set()
    failure=None
    cleanup_confirmed=False
    failure_detail=None
    stage="placement_quote"
    try:
        price_ref=quote(mission)
        stage="independent_watchdog_recheck"
        # Re-observe both independent runners immediately before paid allocation.
        if independent_deadline(mission.marker)!=hard_deadline:
            raise PermissionError("watchdog lease changed")
        stage="independent_reviewer_recheck"
        check_reviewer_active(mission.marker)
        stage="allocation"
        pod,client=allocate(mission,authority,deadline,price_ref,known)
        known.add(pod)
        stage="worker_health"
        health_observations=await_health(client,min(deadline,now()+timedelta(minutes=8)),
            read_provider=lambda:mission.provider_check(pod))
        health_ref=publish(mission.artifacts,{"schema":"anvil.worker-health-observations.v1",
            "pod_id":pod,"observations":health_observations})
        mission.retain(health_ref)
        print("ANVIL_WORKER_HEALTH_REF="+health_ref,flush=True)
        stage="worker_setup"
        if client.execute("setup",mission.source)!={"source_sha256":mission.source_hash,"lock_sha256":LOCK_HASH}:
            raise PermissionError("worker source setup differs")
        stage="cuda_qualification"
        runtime=client.execute("qualify",b"{}")
        runtime_ref=verify_loaded_runtime(runtime,**mission.runtime_args(pod))
        publish(mission.artifacts,runtime["runtime"],runtime_ref)
        mission.retain(runtime_ref)
        mission.mailbox.publish_message("runtime",{"pod_id":pod,"runtime_ref":runtime_ref})
        for slot in range(4):
            stage=f"slot_{slot}"
            check_reviewer_active(mission.marker)
            mission.provider_check(pod)
            task=ctl.prepare_slot(slot)
            prepared=read_verified_json(mission.artifacts,task.prepared_ref)
            mission.retain(task.prepared_ref,prepared["dispatch_payload_ref"],task.checkpoint_ref)
            mission.mailbox.publish_message(f"slot-{slot}-grant-request",{
                "prepared_ref":task.prepared_ref,"payload_ref":prepared["dispatch_payload_ref"],
                "checkpoint_ref":task.checkpoint_ref,"frontier":task.frontier})
            grant,_=wait_message(mission.mailbox,f"slot-{slot}-grant",deadline)
            mission.import_refs(grant["grant_ref"])
            capsule_ref=ctl.reserve_capsule(task,grant_ref=grant["grant_ref"],pod_id=pod,runtime_ref=runtime_ref)
            capsule=read_verified_json(mission.artifacts,capsule_ref)
            mission.retain(capsule_ref)
            slots.append({"slot":slot,"status":"reserved","capsule_ref":capsule_ref})
            result=client.execute(f"slot-{slot}",canonical_bytes(capsule))
            receipt_ref=publish(mission.artifacts,result["document"],result["receipt_ref"])
            slots[-1].update(status="received",receipt_ref=receipt_ref)
            mission.retain(receipt_ref)
            mission.mailbox.publish_message(f"slot-{slot}-receipt-request",{
                "capsule_ref":capsule_ref,"receipt_ref":receipt_ref})
            review,_=wait_message(mission.mailbox,f"slot-{slot}-attestation",deadline)
            mission.import_refs(review["retention_ref"])
            if review["status"]!="verified":
                slots[-1]["status"]="rejected"
                raise RuntimeError("independently rejected role receipt; no retry")
            mission.import_refs(review["attestation_ref"])
            accepted=ctl.admit_slot(slot=slot,receipt_ref=receipt_ref,attestation_ref=review["attestation_ref"],
                retention_ref=review["retention_ref"],retained_artifacts=mission.reader,
                gpu_name=runtime["runtime"]["loads"][0]["gpu_name"])
            slots[-1].update(status="accepted",frontier=accepted.frontier,
                            attestation_ref=review["attestation_ref"],retention_ref=review["retention_ref"])
            mission.checkpoint()
            print("ANVIL_LIVE_ACCEPTED_SLOT="+str(slot),flush=True)
    except Exception as error:
        failure=type(error).__name__
        failure_detail=error.diagnostic if isinstance(error,(PlacementFailure,WorkerHealthFailure)) else safe_failure(stage,error)
        if isinstance(error,WorkerHealthFailure):
            # Preserve startup evidence before destroying the exact owned worker.
            try:
                mission.provider_check(pod)
                failure_detail["startup_logs"]=startup_log_signals(pod,os.environ["RUNPOD_API_KEY"],
                    min(deadline,now()+timedelta(seconds=25)))
            except Exception as log_error:
                failure_detail["startup_log_failure"]=safe_failure("provider_startup_logs",log_error)
        print("ANVIL_LIVE_FAILURE_DETAIL="+json.dumps(failure_detail,sort_keys=True),flush=True)
        print("ANVIL_LIVE_EXECUTION_STOPPED="+failure,flush=True)
    finally:
        # Provider cleanup precedes scientific closure, even on model failure.
        try:
            cleanup_confirmed=mission.identity.cleanup(mission.provider,known_ids=known)
        finally:
            checkpoint=mission.checkpoint()
            packet=build_review_packet(mission.dispatch,slot_records=slots,checkpoint_ref=checkpoint)
            packet.update(execution_failure=failure,execution_failure_detail=failure_detail,primary_cleanup_confirmed=cleanup_confirmed)
            packet_ref=publish(mission.artifacts,packet)
            mission.retain(packet_ref)
            mission.backup_local()
            mission.mailbox.publish_message("terminal-request",{"packet_ref":packet_ref,"checkpoint_ref":checkpoint,
                "slot_records":slots,"primary_cleanup_confirmed":cleanup_confirmed,"known_pod_ids":sorted(known)})
            print("ANVIL_LIVE_REVIEW_PACKET="+packet_ref,flush=True)
    # Reviewer has an independent copy even if this acknowledgement is lost.
    review,_=wait_message(mission.mailbox,"terminal-review",hard_deadline)
    mission.import_refs(review["review_ref"])
    authority.verify(mission.artifacts,review["review_ref"],schema="anvil.execution-review.v1",expected={
        "plan_ref":mission.plan_ref,"packet_ref":packet_ref,"frontier":packet["frontier"],
        "accepted_count":packet["accepted_count"],"independent_restore":True,
        "independent_cleanup_confirmed":True,"semantic_review_pending":True,"outcome":"unresolved",
        "execution_authorized":False})
    mission.repository.append("anvil.live-execution-review",{
        "plan_ref":mission.plan_ref,"review_ref":review["review_ref"],"packet_ref":packet_ref,
        "frontier":packet["frontier"],"scientific_review_pending":True},
        operation_id=review["review_ref"])
    mission.backup_local()
    print("ANVIL_LIVE_EXECUTION_COMPLETE="+str(failure is None and cleanup_confirmed).lower(),flush=True)
    if failure or not cleanup_confirmed:
        raise RuntimeError("bounded first live mission ended unresolved")


def restore_checkpoint(mission,checkpoint_ref,frontier):
    """Restore on a disposable PostgreSQL service, never the hosted Anvil database."""
    import psycopg
    from psycopg.conninfo import conninfo_to_dict
    from hephaestus.state.anvil_postgres import SCHEMA, advance_epoch
    dsn=os.environ["ANVIL_RESTORE_DSN"]
    settings=conninfo_to_dict(dsn)
    if settings.get("host")!="127.0.0.1" or settings.get("dbname")!="anvil_fixture":
        raise PermissionError("independent restore must use the local disposable service")
    def connect():
        return psycopg.connect(dsn)
    with connect() as connection:
        connection.execute(SCHEMA)
        advance_epoch(connection,mission.plan["investigation"],expected_epoch=0)
    restore=PostgresAnvilState(connect,mission.plan["investigation"],1,lambda _:False,lambda *_:False)
    return restore.restore(mission.plan["investigation"],mission.artifacts,checkpoint_ref,expected_frontier=frontier)


def reviewer(mission):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    authority=OffPodAuthority(Ed25519PrivateKey.generate())
    hard_deadline=independent_deadline(mission.marker)
    deadline=hard_deadline-timedelta(seconds=CLEANUP_RESERVE_SECONDS+60)
    mission.mailbox.publish_message("ready",{"activation":mission.marker,"run_id":os.environ["GITHUB_RUN_ID"],
        "plan_ref":mission.plan_ref,"deadline":deadline.isoformat(),
        "public_key_hex":authority.verifier.public_key_hex,"public_key_ref":authority.verifier.public_key_ref})
    terminal=None
    pod=None
    runtime,terminal=wait_message(mission.mailbox,"runtime",hard_deadline,terminal=True)
    if terminal is None:
        pod=runtime["pod_id"]
        intent=mission.mailbox.allocation_intent(mission.plan_ref)
        if (not intent or intent.get("activation")!=mission.marker or intent.get("deadline")!=deadline.isoformat()
                or intent.get("source_sha256")!=mission.source_hash or intent.get("operation_id")!=mission.identity.operation_id):
            raise PermissionError("reviewer cannot confirm exact consumed allocation intent")
        price,=mission.import_refs(intent["price_ref"])
        if (price.get("gpu")!=intent.get("gpu") or price.get("region")!=intent.get("region")
                or price.get("gpu") not in GPU_NAMES or not valid_region(price.get("region"))
                or price.get("schema")!="anvil.first-live-price.v1" or price.get("existing_volume")!=GLOBAL
                or not 0<Decimal(price["all_in_hour_ceiling_usd"])<=MAX_RATE
                or intent.get("maximum_usd")!="0.65" or intent.get("maximum_pods")!=1):
            raise PermissionError("independent placement identity or price differs")
        mission.selected=price
        runtime_ref=runtime["runtime_ref"]
        doc,=mission.import_refs(runtime_ref)
        runtime_result={"runtime":doc,"runtime_ref":runtime_ref}
        for slot in range(4):
            request,terminal=wait_message(mission.mailbox,f"slot-{slot}-grant-request",hard_deadline,terminal=True)
            if terminal is not None:
                break
            mission.provider_check(pod)
            prepared,payload,checkpoint=mission.import_refs(request["prepared_ref"],request["payload_ref"],request["checkpoint_ref"])
            if prepared["dispatch_payload_ref"]!=request["payload_ref"]:
                raise PermissionError("prepared payload ref differs")
            sequence=0 if slot<2 else 2
            expected=mission.dispatch.admission._board.read_snapshot(mission.plan["investigation"],sequence=sequence)
            restored=AnvilState._replay_records(mission.plan["investigation"],checkpoint["records"],request["frontier"])
            if restored!=expected:
                raise PermissionError("independent round checkpoint differs")
            grant_ref=issue_slot(authority,mission.dispatch,prepared_ref=request["prepared_ref"],slot=slot,
                runtime_result=runtime_result,runtime_args=mission.runtime_args(pod),expires_at=deadline)
            mission.retain(grant_ref)
            mission.mailbox.publish_message(f"slot-{slot}-grant",{"grant_ref":grant_ref})
            result,terminal=wait_message(mission.mailbox,f"slot-{slot}-receipt-request",hard_deadline,terminal=True)
            if terminal is not None:
                break
            capsule,receipt=mission.import_refs(result["capsule_ref"],result["receipt_ref"])
            if capsule.get("grant_ref")!=grant_ref or capsule.get("prepared_ref")!=request["prepared_ref"]:
                raise PermissionError("returned capsule differs from issued grant")
            for item in capsule["objects"]:
                mission.import_refs(item["ref"])
            mission.provider_check(pod)
            reviewed=attest_retained_slot(authority,mission.dispatch,slot=slot,capsule_ref=result["capsule_ref"],
                receipt_ref=result["receipt_ref"],runtime_ref=runtime_ref,pod_id=pod,retained_artifacts=mission.reader)
            mission.retain(reviewed["retention_ref"])
            if reviewed["attestation_ref"]:
                mission.retain(reviewed["attestation_ref"])
            mission.mailbox.publish_message(f"slot-{slot}-attestation",reviewed)
            if reviewed["status"]!="verified":
                break
    if terminal is None:
        terminal,_=wait_message(mission.mailbox,"terminal-request",hard_deadline)
    packet,checkpoint=mission.import_refs(terminal["packet_ref"],terminal["checkpoint_ref"])
    for row in terminal["slot_records"]:
        for key in ("receipt_ref","capsule_ref","attestation_ref","retention_ref"):
            if row.get(key):
                mission.import_refs(row[key])
    known=set(terminal["known_pod_ids"])
    if pod:
        known.add(pod)
    if not terminal.get("primary_cleanup_confirmed") or mission.identity._known_live(mission.provider,known):
        raise PermissionError("independent reviewer cannot confirm exact Pod absence")
    snapshot=restore_checkpoint(mission,terminal["checkpoint_ref"],packet["frontier"])
    live=mission.dispatch.admission._board.replay(mission.plan["investigation"])
    if snapshot!=live or packet["accepted_count"]!=len(live.contributions):
        raise PermissionError("independent restoration or accepted frontier differs")
    expected_packet=build_review_packet(mission.dispatch,slot_records=terminal["slot_records"],
                                        checkpoint_ref=terminal["checkpoint_ref"])
    if any(packet.get(k)!=v for k,v in expected_packet.items()):
        raise PermissionError("independent scientific packet differs")
    # Structural review is complete; semantic scores must come from actual
    # independent scientific assessment, not a fabricated deterministic rubric.
    review_ref=authority.sign(mission.artifacts,{"schema":"anvil.execution-review.v1",
        "plan_ref":mission.plan_ref,"packet_ref":terminal["packet_ref"],"frontier":snapshot.frontier,
        "accepted_count":len(snapshot.contributions),"independent_restore":True,
        "independent_cleanup_confirmed":True,"semantic_review_pending":True,"outcome":"unresolved","execution_authorized":False})
    mission.retain(review_ref)
    mission.backup_local()
    mission.mailbox.publish_message("terminal-review",{"review_ref":review_ref})
    print("ANVIL_LIVE_INDEPENDENT_RESTORE_VERIFIED=true",flush=True)
    print("ANVIL_LIVE_INDEPENDENT_REVIEW="+review_ref,flush=True)


def preserve_interrupted_review(mission, error_type):
    """Independent best-effort cold evidence when the primary handoff disappears."""
    checkpoint=mission.checkpoint()
    snapshot=mission.dispatch.admission._board.replay(mission.plan["investigation"])
    rows=[]
    for accepted in snapshot.contributions:
        ref=accepted.origin.inference_ref
        mission.import_refs(ref)
        rows.append({"contribution_id":accepted.contribution_id,"receipt_ref":ref,"frontier":accepted.frontier})
    ref=publish(mission.artifacts,{"schema":"anvil.interrupted-review.v1","plan_ref":mission.plan_ref,
        "checkpoint_ref":checkpoint,"frontier":snapshot.frontier,"accepted":rows,
        "failure_type":error_type,"semantic_review_pending":True,"execution_authorized":False})
    mission.retain(ref)
    mission.backup_local()
    print("ANVIL_INTERRUPTED_REVIEW_EVIDENCE="+ref,flush=True)


if __name__=="__main__":
    mission=None
    mode=None
    try:
        mode=sys.argv[1]
        if mode not in {"controller","reviewer"}:
            raise ValueError("exact controller or reviewer role required")
        mission=Mission(mode)
        (controller if mode=="controller" else reviewer)(mission)
    except Exception as error:
        if mode=="reviewer" and mission is not None:
            try:
                preserve_interrupted_review(mission,type(error).__name__)
            except Exception as recovery_error:
                print("ANVIL_INTERRUPTED_REVIEW_RETENTION_FAILED="+type(recovery_error).__name__,flush=True)
        sys.exit("Anvil bounded live mission stopped ("+type(error).__name__+"); no automatic retry")
