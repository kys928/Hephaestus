"""Scoped single-writer S3 handoff for small first-live evidence objects.

RunPod S3 conditional create is not a multiwriter commit authority. These
content-addressed keys are published by one designated writer per direction,
verified on every read, and must be backed up independently off-provider.
"""
from hashlib import sha256
from io import BytesIO
import re

from hephaestus.storage.verified import canonical_bytes

LIMIT = 2 * 1024 * 1024


class AnvilRetainedReader:
    def __init__(self, client, bucket, activation):
        if not re.fullmatch(r"[0-9a-f]{40}", activation) or bucket != "cviwpryzao":
            raise ValueError("exact first-live retention namespace required")
        self.client, self.bucket = client, bucket
        self.prefix = "hephaestus/anvil/first-live/" + activation

    def _key(self, ref):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", ref):
            raise ValueError("expected content-addressed evidence reference")
        return self.prefix + "/objects/" + ref[7:] + ".json"

    def open(self, ref):
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(ref))
        with response["Body"] as stream:
            raw = stream.read(LIMIT+1)
        if len(raw)>LIMIT or "sha256:"+sha256(raw).hexdigest()!=ref:
            raise ValueError("retained object size or digest differs")
        return BytesIO(raw)


class AnvilRetainedWriter(AnvilRetainedReader):
    def retain(self, ref, artifacts):
        with artifacts.open(ref) as stream:
            raw = stream.read(LIMIT+1)
        if len(raw)>LIMIT or "sha256:"+sha256(raw).hexdigest()!=ref:
            raise ValueError("source evidence differs from retained reference")
        key = self._key(ref)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            code = str(getattr(exc,"response",{}).get("Error",{}).get("Code",""))
            if code not in {"404","NoSuchKey","NotFound"}:
                raise
            self.client.put_object(Bucket=self.bucket, Key=key, Body=raw, ContentType="application/json")
        with self.open(ref) as stream:
            if stream.read()!=raw:
                raise ValueError("retained evidence readback differs")
        return ref

    def publish_message(self, name, document):
        """One writer per fixed message name, never overwrite a different message."""
        if not re.fullmatch(r"(ready|runtime|terminal-request|terminal-review|slot-[0-3]-(grant-request|grant|receipt-request|attestation))",name):
            raise ValueError("unexpected mailbox message")
        raw = canonical_bytes(document)
        if len(raw)>LIMIT:
            raise ValueError("mailbox message exceeds ceiling")
        key = self.prefix + "/messages/" + name + ".json"
        try:
            previous = self.read_message(name)
        except FileNotFoundError:
            self.client.put_object(Bucket=self.bucket,Key=key,Body=raw,ContentType="application/json")
        else:
            if canonical_bytes(previous)!=raw:
                raise PermissionError("mailbox message already differs")
        if canonical_bytes(self.read_message(name))!=raw:
            raise ValueError("mailbox readback differs")

    def read_message(self,name):
        import json
        if not re.fullmatch(r"(ready|runtime|terminal-request|terminal-review|slot-[0-3]-(grant-request|grant|receipt-request|attestation))",name):
            raise ValueError("unexpected mailbox message")
        try:
            response = self.client.get_object(Bucket=self.bucket,Key=self.prefix+"/messages/"+name+".json")
        except Exception as exc:
            code = str(getattr(exc,"response",{}).get("Error",{}).get("Code",""))
            if code in {"404","NoSuchKey","NotFound"}:
                raise FileNotFoundError(name) from None
            raise
        with response["Body"] as stream:
            raw=stream.read(LIMIT+1)
        if len(raw)>LIMIT:
            raise ValueError("mailbox message exceeds ceiling")
        document=json.loads(raw)
        if canonical_bytes(document)!=raw or not isinstance(document,dict):
            raise ValueError("noncanonical mailbox message")
        return document
