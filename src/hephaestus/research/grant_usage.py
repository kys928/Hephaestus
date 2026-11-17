"""Crash-conservative, nonduplicating signed GPU grant ledger.

A durable reservation precedes any model call. No long-lived SQLite/witness
write lock is held across GPU execution, so the existing DurableWorker can
continue its independent heartbeat. A crash after reservation blocks retries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import re

from hephaestus.state.execution_ownership import execution_owner
from hephaestus.storage.verified import canonical_bytes,read_verified_json

_COLLECTION="research_signed_compute_grant_usage"
_DIGEST=re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(slots=True)
class SingleUseComputeGrantLedger:
    repository: object
    artifacts: object

    def run_once(self, *, grant_signature_sha256: str,
                 run_id: str, lineage_id: str, maximum_calls: int,
                 execute: Callable[[],dict]) -> dict:
        if (not _DIGEST.fullmatch(grant_signature_sha256)
                or not run_id or not lineage_id
                or type(maximum_calls) is not int or not 1<=maximum_calls<=6):
            raise PermissionError("signed compute grant accounting identity invalid")

        # Reserve under the SAME kernel-fenced ownership used elsewhere in
        # Hephaestus, then RELEASE before expensive model calls. A separate
        # process observing a reserved grant must stop, never re-invoke models.
        with execution_owner(self.repository,"research-compute-grant",grant_signature_sha256):
            previous=self.repository.get_latest(
                _COLLECTION,"grant_signature_sha256",grant_signature_sha256)
            if previous is not None:
                if (previous.get("run_id")!=run_id
                        or previous.get("lineage_id")!=lineage_id
                        or previous.get("maximum_calls")!=maximum_calls):
                    raise PermissionError("compute grant has been reused for a different study")
                if previous.get("status")=="completed":
                    report=read_verified_json(self.artifacts,str(previous.get("report_ref") or ""))
                    if (report.get("run_id")!=run_id
                            or report.get("lineage_id")!=lineage_id
                            or report.get("role_invocations")!=maximum_calls
                            or report.get("role_inference_scope")!="genuine_certified_frozen_role_receipts"):
                        raise PermissionError("completed grant receipt does not match original science")
                    return report
                raise PermissionError(
                    "compute grant already reserved/failed; new independent authorization required")
            self.repository.append(_COLLECTION,{
                "grant_signature_sha256":grant_signature_sha256,
                "run_id":run_id,"lineage_id":lineage_id,
                "maximum_calls":maximum_calls,"status":"reserved",
            })

        # No program-wide lock spans actual inference: the durable job queue
        # retains its heartbeat and separate ownership. A SIGKILL here leaves
        # the reservation in place and prevents silent double GPU spending.
        try:
            report=execute()
            if (not isinstance(report,dict)
                    or report.get("run_id")!=run_id
                    or report.get("lineage_id")!=lineage_id
                    or report.get("role_invocations")!=maximum_calls
                    or report.get("role_inference_scope")!="genuine_certified_frozen_role_receipts"
                    or report.get("status")!="completed"):
                raise PermissionError("inference did not complete all authorized actual frozen roles")
            ref=self.artifacts.put_bytes(canonical_bytes(report),media_type="application/json")
        except Exception as exc:
            with execution_owner(self.repository,"research-compute-grant",grant_signature_sha256):
                self.repository.append(_COLLECTION,{
                    "grant_signature_sha256":grant_signature_sha256,
                    "run_id":run_id,"lineage_id":lineage_id,
                    "maximum_calls":maximum_calls,"status":"failed_no_automatic_retry",
                    "reason_type":type(exc).__name__,
                })
            raise

        with execution_owner(self.repository,"research-compute-grant",grant_signature_sha256):
            prior=self.repository.get_latest(_COLLECTION,"grant_signature_sha256",
                                             grant_signature_sha256)
            if (prior is None or prior.get("status")!="reserved"
                    or prior.get("run_id")!=run_id or prior.get("lineage_id")!=lineage_id):
                raise PermissionError("grant reservation changed during inference")
            self.repository.append(_COLLECTION,{
                "grant_signature_sha256":grant_signature_sha256,
                "run_id":run_id,"lineage_id":lineage_id,
                "maximum_calls":maximum_calls,"status":"completed",
                "report_ref":ref.artifact_ref,
            })
        return report
