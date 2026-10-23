#!/usr/bin/env python3
"""Read-only A40 boot diagnostics: GraphQL runtime and REST v2 pod log synopsis.

Never writes provider state; never exposes arbitrary log lines or secrets.
"""
import json,os,urllib.request,re,time

POD="bnvazbgs2acrp4"
def runtime():
    query='query { pod(input: {podId: "bnvazbgs2acrp4"}) { id name desiredStatus runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } container { cpuPercent memoryPercent } } } }'
    req=urllib.request.Request("https://api.runpod.io/graphql",
        data=json.dumps({"query":query}).encode(),
        headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=20) as r:data=json.load(r)
    if data.get("errors"):return {"graphql_error":True}
    p=(data.get("data") or {}).get("pod")
    if not isinstance(p,dict):return {"pod_available":False}
    x=p.get("runtime")
    return {"pod_available":True,"desired_status":p.get("desiredStatus"),
        "runtime_available":isinstance(x,dict),
        "uptime_seconds":x.get("uptimeInSeconds") if isinstance(x,dict) else None,
        "gpus":[{"util_pct":g.get("gpuUtilPercent"),"vram_util_pct":g.get("memoryUtilPercent")}
                for g in (x.get("gpus") or []) if isinstance(g,dict)] if isinstance(x,dict) else [],
        "cpu_pct":((x.get("container") or {}).get("cpuPercent") if isinstance(x,dict) else None),
        "mem_pct":((x.get("container") or {}).get("memoryPercent") if isinstance(x,dict) else None)}
def summary_logs():
    u=f"https://api.runpod.io/v2/pods/{POD}/logs?tail=250"
    req=urllib.request.Request(u,headers={
        "Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],
        "Accept":"text/event-stream","User-Agent":"Hephaestus-ReadOnly-A40-LogDiagnostic"})
    out={"api":"runpod_v2_pod_logs","sample_count":0,"container_count":0,
         "system_count":0,"safe_stages":[],"safe_events":[]}
    seen=set()
    with urllib.request.urlopen(req,timeout=16) as r:
        start=time.monotonic()
        while time.monotonic()-start<12 and out["sample_count"]<250:
            try:raw=r.readline(65536)
            except Exception:break
            if not raw:break
            line=raw.decode("utf-8","replace").strip()
            if not line or line.startswith(":"):continue
            if line.startswith("data:"):line=line[5:].strip()
            try: j=json.loads(line)
            except Exception:continue
            if not isinstance(j,dict) or not isinstance(j.get("line"),str):continue
            out["sample_count"]+=1
            src=j.get("source")
            if src=="system":out["system_count"]+=1
            elif src=="container":out["container_count"]+=1
            msg=j["line"]
            ts=j.get("ts")
            if "INTERFACE_REPAIR_V2_BOOTSTRAP_JSON" in msg:
                try:
                    obj=json.loads(msg.split("INTERFACE_REPAIR_V2_BOOTSTRAP_JSON",1)[1].strip())
                    stage=obj.get("stage")
                    if stage and stage not in seen:
                        out["safe_stages"].append({"stage":stage,"timestamp_unix":obj.get("timestamp_unix"),"ts":ts})
                        seen.add(stage)
                except Exception:pass
            # Classify but never emit potentially secret-bearing log lines.
            patterns=(("image_pull",r"(?i)pull.*image|fetching image"),
              ("container_started",r"(?i)start.*container|container.*start"),
              ("repo_checkout",r"(?i)git clone|checking out files|checkout"),
              ("environment_install",r"(?i)pip install|installing collected packages"),
              ("dependency_resolution",r"(?i)collecting |downloading |building wheel"),
              ("gpu_driver",r"(?i)cuda|nvidia-smi|GPU initialized"),
              ("network_failure",r"(?i)connection timed out|TLS handshake|connection refused|name resolution"),
              ("shell_error",r"(?i)traceback|command not found|not found|error:|failed"));
            for k,pattern in patterns:
                if re.search(pattern,msg) and len(out["safe_events"])<45:
                    out["safe_events"].append({"category":k,"ts":ts,"source":src})
                    break
    return out
def main():
    try:print("D_A40_GPU_TELEMETRY "+json.dumps(runtime(),sort_keys=True),flush=True)
    except Exception as e:print("D_A40_GPU_TELEMETRY "+json.dumps({"error_type":type(e).__name__}),flush=True)
    try:print("D_A40_SAFE_BOOTLOG "+json.dumps(summary_logs(),sort_keys=True),flush=True)
    except Exception as e:print("D_A40_SAFE_BOOTLOG "+json.dumps({"error_type":type(e).__name__}),flush=True)
if __name__=="__main__":main()
