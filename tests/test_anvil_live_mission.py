"""CPU-only full handoff rehearsal. No real provider, weights, or live DB."""
from datetime import timedelta
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
from threading import Lock, Thread
from time import sleep

import pytest

from hephaestus.control.anvil_context import AnvilAdmission
from hephaestus.control.anvil_dispatch import AnvilDispatch
from hephaestus.control.anvil_live import publish, publish_frozen_inputs, digest
from hephaestus.control.anvil_live_runtime import MANIFEST
from hephaestus.jobs import SQLiteJobQueue
from hephaestus.production.original_cuda_identity import verified_original_roles
from hephaestus.providers.runpod.anvil_live_retention import AnvilRetainedWriter, AnvilRetainedReader
from hephaestus.schemas.anvil import ContributionOrigin
from hephaestus.state.anvil import AnvilState
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes, read_verified_json

ROOT=Path(__file__).resolve().parents[1]


def test_complete_independent_handoff_without_real_compute(tmp_path,monkeypatch):
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    spec=spec_from_file_location("anvil_mission_fixture",ROOT/"scripts/anvil_live_mission.py")
    mod=module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv("GITHUB_RUN_ID","fixture")
    marker="a"*40
    hard=mod.now()+timedelta(minutes=12)
    monkeypatch.setattr(mod,"independent_deadline",lambda _:hard)
    monkeypatch.setattr(mod,"check_reviewer_active",lambda _:None)
    monkeypatch.setattr(mod,"sleep",lambda _:sleep(.01))
    bundle=json.loads(mod.BUNDLE.read_bytes())
    plan=bundle["plan"]["document"]
    stack,contract=verified_original_roles(plan["registry_ref"])
    material=frozenset(plan["material_refs"])
    scope={k:plan[k] for k in ("investigation","run_id","lineage_id")}
    shared=SQLiteStateRepository(tmp_path/"shared.sqlite")
    image={"schema":"anvil.image-dependency-qualification.v1","python":"3.12.3",
           "torch":"2.8.0+cu128","cuda":"12.8","packages":{"torch":"2.8.0+cu128"},
           "lock_sha256":mod.LOCK_HASH,"cuda_execution":False}

    class Missing(Exception):
        response={"Error":{"Code":"NoSuchKey"}}
    class S3:
        def __init__(self):
            self.data={}
            self.lock=Lock()
        def head_object(self,*,Bucket,Key):
            with self.lock:
                if (Bucket,Key) not in self.data:
                    raise Missing()
        def get_object(self,*,Bucket,Key):
            self.head_object(Bucket=Bucket,Key=Key)
            with self.lock:
                return {"Body":BytesIO(self.data[Bucket,Key])}
        def put_object(self,*,Bucket,Key,Body,**kw):
            with self.lock:
                self.data[Bucket,Key]=Body
    s3=S3()
    closed=[]
    class Identity:
        operation_id = "b"*64
        def cleanup(self,provider,*,known_ids):
            closed.extend(known_ids)
            return True
        def _known_live(self,provider,known):
            assert "fixture123" in closed
            return []
    class FixtureMission(mod.Mission):
        def __init__(self,mode):
            self.marker=marker
            self.root=tmp_path/mode
            self.root.mkdir()
            self.artifacts=FileSystemArtifactStore(self.root/"artifacts")
            self.repository=SQLiteStateRepository(self.root/"state.sqlite")
            self.queue=SQLiteJobQueue(self.root/"jobs.sqlite")
            self.bundle=bundle
            self.plan=plan
            self.material=material
            self.plan_ref=publish_frozen_inputs(bundle,self.artifacts)
            self.source=b"fixture source bytes"
            self.source_hash=sha256(self.source).hexdigest()
            self.image_qualification=image
            self.provider=object()
            self.identity=Identity()
            self.mailbox=AnvilRetainedWriter(s3,"cviwpryzao",marker)
            self.reader=AnvilRetainedReader(s3,"cviwpryzao",marker)
            self.dispatch=self.make_dispatch(frozenset())
            role=stack.resolve("diagnosis")
            origin=ContributionOrigin(**scope,role="diagnosis",inference_ref=self.plan_ref,
                context_frontier=bundle["expected_initial_frontier"],idempotency_key="material",
                model_id=role.model_id,model_revision=role.revision,adapter_ref="sha256:"+role.adapter.sha256)
            for ref in material:
                self.dispatch.admission.admit_material(origin,ref)
        def make_dispatch(self,trusted):
            board=AnvilState(shared,lambda origin:origin.inference_ref in trusted,lambda origin,ref:ref in material)
            return AnvilDispatch(AnvilAdmission(board,self.artifacts,plan["registry_ref"],
                material,trusted,self.repository),self.queue,self.plan_ref)
        def provider_check(self,pod_id):
            assert pod_id=="fixture123"
    primary=FixtureMission("primary")
    reviewer=FixtureMission("reviewer")
    def quote(mission):
        doc={"schema":"anvil.first-live-price.v1","gpu":"NVIDIA RTX A6000","region":"CA-MTL-1",
             "existing_volume":mod.GLOBAL,"all_in_hour_ceiling_usd":"0.60"}
        mission.selected=doc
        ref=publish(mission.artifacts,doc)
        mission.retain(ref)
        return ref
    monkeypatch.setattr(mod,"quote",quote)
    def restore(mission,ref,frontier):
        board=AnvilState(SQLiteStateRepository(mission.root/"restored.sqlite"),lambda _:False,lambda *_:False)
        return board.restore(plan["investigation"],mission.artifacts,ref,expected_frontier=frontier)
    monkeypatch.setattr(mod,"restore_checkpoint",restore)
    contexts=[]
    class Worker:
        def request(self,path):
            assert path=="/health"
            return {"status":"ready"}
        def execute(self,name,raw):
            if name=="setup":
                return {"source_sha256":primary.source_hash,"lock_sha256":mod.LOCK_HASH}
            if name=="qualify":
                env={k:image[k] for k in ("python","torch","cuda")}
                env["packages_sha256"]=sha256(canonical_bytes(image["packages"])).hexdigest()
                loads=[]
                for role in ("diagnosis","planner"):
                    model=stack.resolve(role)
                    loads.append({"schema":"anvil.cuda-offline-load.v1","pod_id":"fixture123","role":role,
                        "model_id":model.model_id,"revision":model.revision,"adapter_sha256":model.adapter.sha256,
                        "manifest_sha256":MANIFEST,"cuda":"12.8","environment":env,
                        "generation_count":0,"peak_allocated_bytes":100,"gpu_sku":"NVIDIA RTX A6000","gpu_name":"NVIDIA RTX A6000","vram_bytes":48*1024**3,"compute_capability":[8,6]})
                runtime={"schema":"anvil.qualified-runtime.v1","pod_id":"fixture123",
                    "source_sha256":primary.source_hash,"lock_sha256":mod.LOCK_HASH,"loads":loads}
                return {"runtime":runtime,"runtime_ref":digest(runtime)}
            capsule=json.loads(raw)
            prepared=read_verified_json(primary.artifacts,capsule["prepared_ref"])
            payload=read_verified_json(primary.artifacts,prepared["dispatch_payload_ref"])
            task=primary.dispatch._prepare(payload)
            contexts.append(json.loads(task.situation))
            model=stack.resolve(task.role)
            vocab=contract.profile(task.role)["vocabulary"]
            answer={k:vocab[k][0] for k in ("decision","action","primary_variable")}
            answer.update(confidence=.5,uncertainties=["CPU fixture only"],evidence_refs=[task.evidence[0].ref],
                          rationale="Synthetic integration fixture, never scientific evidence.")
            output=canonical_bytes(answer).decode()
            prompt=contract.prompt(task)
            receipt=dict(schema="role-inference.v1",status="accepted",role=task.role,run_id=task.run_id,
                lineage_id=task.lineage_id,stage_name=task.stage_name,raw_output=output,answer=answer,
                output_sha256=sha256(output.encode()).hexdigest(),prompt=prompt,
                prompt_sha256=sha256(prompt.encode()).hexdigest(),model_id=model.model_id,
                base_revision=model.revision,adapter_sha256=model.adapter.sha256,adapter_source=model.adapter.s3_key,
                tokenizer_id=model.model_id,tokenizer_revision=model.revision,evidence_refs=[e.ref for e in task.evidence],
                generator="PinnedTransformersRoleGenerator",seed=271828,max_new_tokens=512,
                execution={"device":"cuda:0","gpu_name":"NVIDIA RTX A6000","cuda_version":"12.8","generated_token_ids":[1,2]})
            return {"receipt_ref":digest(receipt),"document":receipt}
    def allocate(mission,authority,deadline,price_ref,known):
        mission.mailbox.consume_allocation(mission.plan_ref,{
            "schema":"anvil.first-live-allocation-intent.v1","plan_ref":mission.plan_ref,
            "activation":mission.marker,"source_sha256":mission.source_hash,"price_ref":price_ref,
            "operation_id":mission.identity.operation_id,"deadline":deadline.isoformat(),"gpu":"NVIDIA RTX A6000","region":"CA-MTL-1","maximum_usd":"0.65","maximum_pods":1})
        known.add("fixture123")
        return "fixture123",Worker()
    monkeypatch.setattr(mod,"allocate",allocate)
    errors=[]
    def run(which,mission):
        try:
            which(mission)
        except Exception as exc:
            errors.append(exc)
    # Daemon threads keep a failed fixture from trapping the test process.
    threads=[Thread(target=run,args=(fn,m),daemon=True) for fn,m in [(mod.reviewer,reviewer),(mod.controller,primary)]]
    for t in threads:t.start()
    for t in threads:t.join(20)
    assert not errors, repr(errors)
    assert not any(t.is_alive() for t in threads)
    assert len(contexts)==4 and contexts[0]==contexts[1] and contexts[2]==contexts[3]
    assert len(contexts[2]["advisory_history"])==2
    assert closed==["fixture123"]
    snap=primary.dispatch.admission._board.replay(plan["investigation"])
    assert len(snap.contributions)==4
    review=primary.repository.all("anvil.live-execution-review")
    assert len(review)==1 and review[0]["scientific_review_pending"] is True
