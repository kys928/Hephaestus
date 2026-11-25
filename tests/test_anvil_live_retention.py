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
