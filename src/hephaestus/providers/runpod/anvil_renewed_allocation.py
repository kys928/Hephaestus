"""Exact, separately approved second allocation; never reset original consumption.

This is a single new authorization, not a new scientific investigation or an
activation-SHA-derived retry budget. The first approval remains immutable.
"""
from hashlib import sha256
import json

from hephaestus.providers.runpod.anvil_live_retention import AnvilRetainedWriter
from hephaestus.providers.runpod.anvil_live_lease import AnvilPodIdentity
from hephaestus.storage.verified import canonical_bytes

APPROVAL = "operator-20261125-second-one-gpu-0.65-usd-3600s"
PLAN = "sha256:a7d8d0e9bf44fec0981b5639fc3bce1d735865ed9fa0ab44256df6c6b63e1950"
PREVIOUS_ACTIVATION = "c280e97b5e917a638429bbb2889403290020767f"
PREVIOUS_POD = "rfjuzxnbi0qr60"
PREVIOUS_INTENT_SHA = "b69eb8b76d6e026a2fb823bbaae994bf5b185a02674d15ae3c8a92aedea4b8e9"
PREVIOUS_PACKET = "sha256:a2370fc722314d12307be87debd9f6ab2f748e62d920b904c6d44e388b07312b"
PREVIOUS_REVIEW = "sha256:285d1f0edfe8ccbb6a96d9880e8ffcc9ce7c133705a2c236ee899a0a33aaf19b"


class RenewedAnvilAllocation(AnvilRetainedWriter):
    def __init__(self, client, bucket, activation, approval):
        super().__init__(client,bucket,activation)
        if approval != APPROVAL or activation == PREVIOUS_ACTIVATION:
            raise PermissionError("exact separate operator allocation authorization required")
        self.previous = AnvilRetainedWriter(client,bucket,PREVIOUS_ACTIVATION)

    def _allocation_key(self, plan_ref):
        if plan_ref != PLAN:
            raise PermissionError("renewed allocation is bound to the original frozen plan")
        return super()._allocation_key(plan_ref).removesuffix(".json") + "/" + APPROVAL + ".json"

    def verify_previous(self, provider):
        old = self.previous.allocation_intent(PLAN)
        if not old or sha256(canonical_bytes(old)).hexdigest() != PREVIOUS_INTENT_SHA:
            raise PermissionError("original consumed allocation differs")
        terminal = self.previous.read_message("terminal-request")
        review = self.previous.read_message("terminal-review")
        if (terminal.get("known_pod_ids") != [PREVIOUS_POD]
                or terminal.get("primary_cleanup_confirmed") is not True
                or terminal.get("packet_ref") != PREVIOUS_PACKET
                or review.get("review_ref") != PREVIOUS_REVIEW):
            raise PermissionError("original terminal evidence differs")
        with self.previous.open(PREVIOUS_PACKET) as stream:
            packet = json.load(stream)
        with self.previous.open(PREVIOUS_REVIEW) as stream:
            json.load(stream)  # content-address verified; original review previously independently verified
        if packet.get("accepted_count") != 0 or packet.get("primary_cleanup_confirmed") is not True:
            raise PermissionError("prior investigation not empty and closed")
        if AnvilPodIdentity(PREVIOUS_ACTIVATION)._known_live(provider,{PREVIOUS_POD}):
            raise PermissionError("previous exact Pod not absent; no further allocation")

    def allocation_intent(self, plan_ref):
        doc=super().allocation_intent(plan_ref)
        if doc is not None and (doc.get("approval") != APPROVAL or
                doc.get("previous_intent_sha256") != PREVIOUS_INTENT_SHA):
            raise PermissionError("renewed allocation intent lacks exact approval binding")
        return doc

    def consume_allocation(self, plan_ref, document):
        if document.get("maximum_usd") != "0.65" or document.get("maximum_pods") != 1:
            raise PermissionError("renewed allocation exceeds one-shot scope")
        return super().consume_allocation(plan_ref,dict(document,approval=APPROVAL,
            previous_intent_sha256=PREVIOUS_INTENT_SHA))
