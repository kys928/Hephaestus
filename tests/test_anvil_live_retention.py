from io import BytesIO
import pytest
from hephaestus.control.anvil_live import publish
from hephaestus.providers.runpod.anvil_live_retention import AnvilRetainedReader, AnvilRetainedWriter
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import read_verified_json


class Missing(Exception):
    response={"Error":{"Code":"NoSuchKey"}}


class S3:
    def __init__(self):
        self.objects={}
    def head_object(self, *, Bucket, Key):
        if (Bucket,Key) not in self.objects:
            raise Missing()
    def get_object(self, *, Bucket, Key):
        self.head_object(Bucket=Bucket,Key=Key)
        return {"Body":BytesIO(self.objects[Bucket,Key])}
    def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.objects[Bucket,Key]=Body


def test_scoped_handoff_and_corruption(tmp_path):
    client=S3()
    writer=AnvilRetainedWriter(client,"cviwpryzao","a"*40)
    reader=AnvilRetainedReader(client,"cviwpryzao","a"*40)
    store=FileSystemArtifactStore(tmp_path)
    ref=publish(store,{"fixture":True})
    writer.retain(ref,store)
    assert read_verified_json(reader,ref)=={"fixture":True}
    writer.publish_message("ready",{"ref":ref})
    writer.publish_message("ready",{"ref":ref})
    with pytest.raises(PermissionError):
        writer.publish_message("ready",{"ref":"different"})
    with pytest.raises(ValueError):
        writer.publish_message("../escape",{})
    client.objects[writer.bucket,writer._key(ref)]=b"{}"
    with pytest.raises(ValueError):
        read_verified_json(reader,ref)


def test_approval_consumed_across_activation_commits():
    client=S3()
    plan="sha256:"+"f"*64
    a=AnvilRetainedWriter(client,"cviwpryzao","a"*40)
    b=AnvilRetainedWriter(client,"cviwpryzao","b"*40)
    doc={"schema":"anvil.first-live-allocation-intent.v1","plan_ref":plan,"activation":"a"*40}
    assert a.allocation_intent(plan) is None
    a.consume_allocation(plan,doc)
    assert b.allocation_intent(plan)==doc
    with pytest.raises(PermissionError,match="already consumed"):
        b.consume_allocation(plan,dict(doc,activation="b"*40))

def test_renewal_is_separate_fixed_one_use_record(monkeypatch):
    import hephaestus.providers.runpod.anvil_renewed_allocation as renewal
    from hephaestus.storage.verified import canonical_bytes
    from hashlib import sha256
    client=S3()
    old=AnvilRetainedWriter(client,"cviwpryzao",renewal.PREVIOUS_ACTIVATION)
    doc={"schema":"anvil.first-live-allocation-intent.v1","plan_ref":renewal.PLAN,
         "activation":renewal.PREVIOUS_ACTIVATION}
    old.consume_allocation(renewal.PLAN,doc)
    monkeypatch.setattr(renewal,"PREVIOUS_INTENT_SHA",sha256(canonical_bytes(doc)).hexdigest())
    a=renewal.RenewedAnvilAllocation(client,"cviwpryzao","a"*40,renewal.APPROVAL)
    b=renewal.RenewedAnvilAllocation(client,"cviwpryzao","b"*40,renewal.APPROVAL)
    assert a.allocation_intent(renewal.PLAN) is None
    with pytest.raises(PermissionError):
        a.allocation_intent("sha256:"+"f"*64)
    with pytest.raises(PermissionError):
        renewal.RenewedAnvilAllocation(client,"cviwpryzao","a"*40,"unapproved")
    new=dict(doc,activation="a"*40,maximum_usd="0.65",maximum_pods=1)
    a.consume_allocation(renewal.PLAN,new)
    assert b.allocation_intent(renewal.PLAN)["approval"]==renewal.APPROVAL
    with pytest.raises(PermissionError,match="already consumed"):
        b.consume_allocation(renewal.PLAN,dict(new,activation="b"*40))
    assert old.allocation_intent(renewal.PLAN)==doc


def test_renewal_requires_exact_original_terminal_evidence():
    import hephaestus.providers.runpod.anvil_renewed_allocation as renewal
    a=renewal.RenewedAnvilAllocation(S3(),"cviwpryzao","a"*40,renewal.APPROVAL)
    with pytest.raises(PermissionError,match="original consumed"):
        a.verify_previous(None)
