from __future__ import annotations
import argparse, json
from pathlib import Path
from hephaestus.state.run_store import RunStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.manifest_store import ManifestStore
from hephaestus.state.report_store import ReportStore
from hephaestus.state.memory_store import MemoryStore
from hephaestus.state.artifact_index import ArtifactIndex


def _load(state_root: Path, run_id: str) -> dict[str, object]:
    run = RunStore(state_root).get(run_id)
    if not run:
        raise ValueError(f"run_id '{run_id}' not found")
    warnings=[]
    lineage = LineageStore(state_root).get_current(str(run.get("lineage_id") or "")) if run.get("lineage_id") else None
    if not lineage: warnings.append("missing_lineage")
    manifests = ManifestStore(state_root)
    manifest = manifests.get(str(run.get("data_manifest_id") or "")) if run.get("data_manifest_id") else None
    if not manifest:
        lst = manifests.list_for_run(run_id)
        manifest = lst[-1] if lst else None
    if not manifest: warnings.append("missing_manifest")
    reports=[r for r in ReportStore(state_root).all() if str(r.get("kind"))=="eval_report" and str(r.get("run_id"))==run_id]
    eval_report = reports[-1] if reports else None
    if not eval_report: warnings.append("missing_eval")
    decision = DecisionStore(state_root).get(f"dec-{run_id}-exit")
    if decision and "effective_action" not in decision:
        decision["effective_action"] = str(((decision.get("metadata") or {}).get("effective_action") or ""))
    if not decision: warnings.append("missing_decision")
    memory = MemoryStore(state_root).list_for_run(run_id)
    if not memory: warnings.append("missing_memory")
    artifacts = [a for a in ArtifactIndex(state_root).all() if str(a.get("run_id") or "") == run_id]
    replay = dict((((decision or {}).get("metadata") or {}).get("replay_requirements") or {}))
    if not replay and isinstance(eval_report, dict):
        replay = dict(eval_report.get("replay_metadata") or {})
    replay.setdefault("replay_scope", "unknown")
    return {"run":run,"lineage":lineage,"manifest":manifest,"eval":eval_report,"decision":decision,"replay":replay,"memory":memory,"artifacts":artifacts,"warnings":warnings}


def _text(data: dict[str, object], no_color: bool=False) -> str:
    return "\n".join([
        "HEPHAESTUS RUN INSPECTION",
        "Run",f"run_id: {data['run'].get('run_id')}",
        "Lineage","Data Manifest","Evaluation","Decision / Gates","Replay","Memory","Artifacts","Warnings",
    ])

def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(); p.add_argument('--state-root', required=True); p.add_argument('--run-id', required=True); p.add_argument('--format', choices=['text','json'], default='text'); p.add_argument('--no-color', action='store_true')
    args=p.parse_args(argv)
    root=Path(args.state_root)
    if not root.exists():
        print(f"invalid state_root: {root}")
        return 2
    try:
        data=_load(root,args.run_id)
    except ValueError as exc:
        print(str(exc)); return 1
    if args.format=='json': print(json.dumps(data, indent=2, sort_keys=True))
    else: print(_text(data,args.no_color))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
