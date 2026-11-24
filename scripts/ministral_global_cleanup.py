"""Exact-campaign provider termination independent of local SQLite admission.

The operator controller uses direct REST creation for Global mounts. Its
emergency deletion must not depend on an unrelated SQLite operation ledger.
"""
import re,time
from hephaestus.providers.runpod.execution import RunPodApiError

def terminate_campaign_pods(adapter, run_tag, *, polls=12, sleep=time.sleep):
    if not re.fullmatch(r"[0-9]{6,20}",run_tag):
        raise ValueError("exact campaign identity required")
    prefix="hephaestus-ministral-triad-"+run_tag+"-global-"
    pattern=re.compile(re.escape(prefix)+r"[1-6]\Z")
    def owned():
        return [p for p in adapter.list_pods()
                if str(p.get("name","")).startswith(prefix)]
    for row in owned():
        ident=row.get("id")
        if (not pattern.fullmatch(str(row.get("name","")))
                or not isinstance(ident,str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{4,90}",ident)):
            raise PermissionError("unexpected campaign resource identity")
        current=adapter.get_pod(ident)
        if current.get("id")!=ident or current.get("name")!=row.get("name"):
            raise PermissionError("provider identity changed before termination")
        try:
            status,_=adapter._request("DELETE","pods/"+ident)
        except RunPodApiError as exc:
            if exc.status_code!=404:raise
            status=404
        if status not in (200,202,204,404):
            raise RuntimeError("provider termination not acknowledged")
    for _ in range(polls):
        if not owned():return
        sleep(5)
    raise RuntimeError("campaign Pod still present after bounded cleanup")
