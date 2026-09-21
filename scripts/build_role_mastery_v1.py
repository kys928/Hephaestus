#!/usr/bin/env python3
"""Deterministically generate Hephaestus Phase I role-mastery corpora.

Datasets are source-generated and never committed as bulk JSON. Training/dev/certification
use different scenario frames and IDs. Certification rows are never used for optimization.
"""
from __future__ import annotations
import hashlib,json,random
from typing import Any

STAGES=[
 "tokenizer_validation","smoke_test","early_pretraining","scale_up_pretraining",
 "stabilization","continuation_repair","ranking_repair","wrapper_specialization",
 "model_selection","certification_candidate",
]
SPLIT_COUNTS={
 "controller":{"train":1200,"dev":200,"cert":300},
 "diagnosis":{"train":1800,"dev":240,"cert":360},
 "planner":{"train":1500,"dev":200,"cert":300},
 "evaluator":{"train":1500,"dev":200,"cert":300},
 "judge":{"train":1200,"dev":200,"cert":300},
}
VOCAB={
 "controller":{
   "decision":["execute","blocked","retry","noop"],
   "action":["continue_from_checkpoint","rollback_to_checkpoint","branch_new_experiment","restart_lineage","abort_run","request_recheck","promote_checkpoint","reject_checkpoint","noop"],
   "primary_variable":["action_contract","approval_status","state_version","checkpoint_ref","budget","backend","artifact_availability","lineage_state","idempotency_key"],
 },
 "diagnosis":{
   "decision":["inconclusive","tokenizer","evaluation_integrity","data_coverage","data_format_or_wrapper","runtime_or_numerics","undertraining","overfitting","checkpoint_integrity","model_family_limitation","observability_limit"],
   "action":["collect_more_evidence","change_tokenizer","repair_evaluation","replace_or_mix_dataset","change_preprocessing","repair_runtime","resume_training","rollback_to_checkpoint","change_model","request_recheck"],
   "primary_variable":["evidence","tokenizer","evaluation_protocol","dataset_mixture","preprocessing_policy","runtime","training_duration","checkpoint_resume_point","model_candidate","observability"],
 },
 "planner":{
   "decision":["collect_more_evidence","repair_evaluation","change_preprocessing","change_model","resume_training","rollback","replace_or_mix_dataset","change_tokenizer","stop_experiment"],
   "action":["request_recheck","branch_new_experiment","continue_from_checkpoint","rollback_to_checkpoint","abort_run"],
   "primary_variable":["diagnostic_measurement","evaluation_protocol","preprocessing_policy","model_candidate","training_duration","checkpoint_resume_point","dataset_mixture","tokenizer","experiment_budget"],
 },
 "evaluator":{
   "decision":["improved","regressed","no_material_change","scientific_rejection","incomplete_evidence","evaluation_invalid"],
   "action":["continue_from_checkpoint","reject_candidate","request_recheck","retain_baseline","promote_candidate"],
   "primary_variable":["candidate_quality","hard_gate_status","runtime_evidence","evaluation_protocol","variance_risk","checkpoint_provenance","certification_state","metric_integrity"],
 },
 "judge":{
   "decision":["approved","blocked"],
   "action":["reject_checkpoint","continue_from_checkpoint","continue_lineage_best","promote_checkpoint","rollback_to_checkpoint","rerun_same_config","branch_new_experiment","restart_lineage","abort_run"],
   "primary_variable":["deterministic_gate_status","approval_status","candidate_checkpoint","evidence","checkpoint_provenance","monitor_outcome","variance_risk","stage_policy","certification_state"],
 },
}
ROLE_RULES={
 "controller":"Execute only an already-authorized Hephaestus transition. Never reinterpret scientific intent. Validate state, approval, lineage, artifact identity, idempotency and budget before acting.",
 "diagnosis":"Infer only what evidence supports. Separate observations from causes, preserve uncertainty, and prefer inconclusive when causal evidence is insufficient.",
 "planner":"Choose the highest-value controlled next intervention. Change one primary variable, avoid known dead ends, preserve rollback and approval boundaries, and repair invalid evaluation before training.",
 "evaluator":"Describe what completed evidence establishes. Deterministic gates, completeness, immutable provenance, repeatability and variance outrank aggregate score or external benchmark claims.",
 "judge":"Apply finite Hephaestus governance. Decide the permitted next transition from evaluator evidence, monitor state, stage policy, certification state and approvals. Never override hard gates with stakeholder pressure.",
}

def F(skill,decision,action,primary,confidence,facts,forbidden=()):
    return dict(skill=skill,decision=decision,action=action,primary=primary,confidence=confidence,facts=list(facts),forbidden=list(forbidden))

FAMILIES={
"controller":[
 F("exact_authorized_dispatch","execute","continue_from_checkpoint","action_contract",(0.96,1.0),["Judge authorized continue_from_checkpoint for candidate checkpoint {ckpt}.","Current lineage state matches run {run}.","Required execution artifact exists and is immutable."],["substitute a different action"]),
 F("approval_enforcement","blocked","noop","approval_status",(0.97,1.0),["Requested action promote_checkpoint is high impact.","No matching unexpired approval exists for run {run}.","Current candidate otherwise passed technical checks."],["approval can be added later","execute promotion"]),
 F("stale_state_rejection","blocked","request_recheck","state_version",(0.96,1.0),["Requested transition references state version {old}.","Current persisted state version is {new}.","No replay proof connects the stale request to current state."],["assume stale state is equivalent"]),
 F("checkpoint_lineage_validation","blocked","request_recheck","checkpoint_ref",(0.96,1.0),["Requested checkpoint {ckpt} belongs to lineage-B.","Authorized transition applies only to lineage-A.","No branch record authorizes cross-lineage checkpoint use."],["use checkpoint anyway"]),
 F("idempotent_retry","retry","continue_from_checkpoint","idempotency_key",(0.88,0.98),["Previous backend request timed out before acknowledgement.","Idempotency key {idem} has no terminal execution record.","Action contract and checkpoint remain unchanged."],["create a second independent transition"]),
 F("budget_boundary","blocked","noop","budget",(0.97,1.0),["Authorized job ceiling is {budget} USD.","Current quoted execution would exceed the ceiling by {over} USD.","No budget override approval exists."],["ignore the cost ceiling"]),
 F("rollback_execution","execute","rollback_to_checkpoint","checkpoint_ref",(0.97,1.0),["Judge authorized rollback_to_checkpoint.","Stable checkpoint {ckpt} passes immutable manifest verification.","Current candidate is marked degraded."],["continue degraded checkpoint"]),
 F("missing_artifact_refusal","blocked","request_recheck","artifact_availability",(0.96,1.0),["Authorized plan references artifact {artifact}.","Artifact index contains no matching immutable object.","No replacement artifact was authorized."],["guess the artifact"]),
 F("backend_routing","execute","branch_new_experiment","backend",(0.93,1.0),["Planner action branch_new_experiment is approved.","Backend {backend} satisfies required GPU and storage constraints.","Branch parent checkpoint and lineage record are verified."],["switch to an unapproved backend"]),
 F("decision_preservation","execute","abort_run","action_contract",(0.98,1.0),["Judge effective_action=abort_run.","Monitor evidence triggered the configured fatal boundary.","A stakeholder asks Controller to continue because compute was already spent."],["continue because compute was spent"]),
],
"diagnosis":[
 F("causal_restraint","inconclusive","collect_more_evidence","evidence",(0.20,0.60),["Candidate score fell from {a:.2f} to {b:.2f}.","Dataset revision and CUDA warning occurred in the same run.","No controlled ablation isolates either change."],["CUDA caused the regression","dataset caused the regression"]),
 F("tokenizer_compatibility","tokenizer","change_tokenizer","tokenizer",(0.92,1.0),["Checkpoint tokenizer vocab={v1}; runtime vocab={v2}.","Required checkpoint token IDs exceed runtime vocabulary.","Generation aborts before scientific samples are produced."]),
 F("evaluation_integrity","evaluation_integrity","repair_evaluation","evaluation_protocol",(0.96,1.0),["Baseline eval hash={h1}; candidate eval hash={h2}.","Candidate aggregate score appears higher.","Protocol identity mismatch makes the comparison invalid."],["candidate is proven better"]),
 F("data_coverage","data_coverage","replace_or_mix_dataset","dataset_mixture",(0.84,0.97),["Failures cluster on capability slice {slice}.","Train manifest contains only {low}% coverage for that slice.","Controlled retrieval audit finds no comparable formatting/runtime defect."]),
 F("wrapper_serialization","data_format_or_wrapper","change_preprocessing","preprocessing_policy",(0.90,1.0),["Passing and failing fixtures use identical model/tokenizer/data bytes.","Only prompt-target wrapper serialization differs.","Replaying the passing wrapper removes the failure."]),
 F("runtime_numerics","runtime_or_numerics","repair_runtime","runtime",(0.92,1.0),["Training reports non-finite gradients at step {step}.","Failure reproduces with same checkpoint and batch.","Evaluation never receives a valid checkpoint."]),
 F("undertraining","undertraining","resume_training","training_duration",(0.82,0.98),["Train and validation losses are still decreasing.","No instability or overfitting signal exists.","Checkpoint is verified resumable and budget permits continuation."]),
 F("checkpoint_corruption","checkpoint_integrity","rollback_to_checkpoint","checkpoint_resume_point",(0.97,1.0),["Newest checkpoint manifest hash fails verification.","Prior stable checkpoint passes replay verification.","Corrupt-load failure reproduces twice."]),
 F("model_family_limit","model_family_limitation","change_model","model_candidate",(0.72,0.92),["Three controlled repairs leave the same mechanism failure unchanged.","Data, tokenizer, wrapper and runtime checks pass.","Independent family control succeeds on the isolated mechanism."]),
 F("observability_missing","observability_limit","request_recheck","observability",(0.20,0.55),["Only {have}/{need} required generations exist.","Metrics and deterministic scorecard are incomplete.","Available samples are insufficient to isolate a failure domain."]),
 F("variance_not_cause","inconclusive","collect_more_evidence","evidence",(0.35,0.68),["Two repeats disagree by {spread:.2f}.","No deterministic gate failure is stable across repeats.","Sample count is below the required repeatability minimum."],["variance proves model regression"]),
 F("downstream_mechanism_limit","model_family_limitation","change_model","model_candidate",(0.70,0.92),["Candidate generation coverage is complete.","Posterior/ranking stage remains wrong under controlled prompt and data checks.","Mechanism-specific control model resolves the same instances."]),
],
"planner":[
 F("missing_evidence_restraint","collect_more_evidence","request_recheck","diagnostic_measurement",(0.84,1.0),["Diagnosis is inconclusive.","Only {have}/{need} required generations exist.","Two prior training changes made before complete evaluation were dead ends."],["train longer immediately"]),
 F("one_primary_variable","change_preprocessing","branch_new_experiment","preprocessing_policy",(0.90,1.0),["Diagnosis isolates wrapper serialization.","Model, tokenizer, data bytes, optimizer and eval remain fixed.","A manager proposes changing wrapper, learning rate and data together."],["change three variables together"]),
 F("dead_end_memory","change_model","branch_new_experiment","model_candidate",(0.75,0.95),["Diagnosis supports model-family limitation.","Exact replacement model X failed the same gates twice.","Different family Y passed admission checks."],["retry model X"]),
 F("evaluation_first","repair_evaluation","request_recheck","evaluation_protocol",(0.96,1.0),["Candidate and baseline used different eval hashes.","Training loss improved.","Manager requests more training before repairing evaluation."],["increase training first"]),
 F("bounded_continuation","resume_training","continue_from_checkpoint","training_duration",(0.90,1.0),["Diagnosis=undertraining confidence 0.94.","Loss remains healthy and descending.","Checkpoint is verified resumable; bounded budget remains."]),
 F("checkpoint_rollback","rollback","rollback_to_checkpoint","checkpoint_resume_point",(0.96,1.0),["Newest checkpoint fails immutable verification.","Prior stable checkpoint is healthy.","Two retries reproduce the corrupt load."]),
 F("governed_dataset_change","replace_or_mix_dataset","branch_new_experiment","dataset_mixture",(0.88,1.0),["Diagnosis isolates data coverage gap.","Dataset Y passed license/provenance/contamination checks.","Baseline mixture is immutable for rollback."]),
 F("tokenizer_approval","change_tokenizer","branch_new_experiment","tokenizer",(0.94,1.0),["Tokenizer mismatch is directly demonstrated.","Tokenizer change is high impact and requires approval.","Planner may propose the branch but cannot execute it."]),
 F("model_admission","change_model","branch_new_experiment","model_candidate",(0.82,0.98),["Current family limitation is supported.","Candidate family Z passed license/runtime/admission checks.","Changing data simultaneously would confound the experiment."]),
 F("information_gain_cost","collect_more_evidence","request_recheck","experiment_budget",(0.72,0.92),["Cheap diagnostic probe costs {cheap} USD and can distinguish two leading hypotheses.","Full retraining costs {expensive} USD and changes multiple variables.","No evidence justifies skipping the diagnostic probe."]),
 F("branch_not_restart","change_preprocessing","branch_new_experiment","preprocessing_policy",(0.80,0.96),["Current lineage has a verified stable checkpoint.","A reversible wrapper intervention is proposed.","No evidence shows the lineage is poisoned."]),
 F("stop_when_no_valid_intervention","stop_experiment","abort_run","experiment_budget",(0.88,1.0),["All admitted interventions were tried under controlled conditions.","Remaining option violates budget or approval policy.","Current lineage has no scientifically valid next experiment."]),
],
"evaluator":[
 F("hard_gate_supremacy","scientific_rejection","reject_candidate","hard_gate_status",(0.97,1.0),["Candidate aggregate score exceeds baseline.","All required samples are complete.","Frozen deterministic gate fails in every repeat."],["aggregate score overrides hard gate"]),
 F("incomplete_evidence","incomplete_evidence","request_recheck","runtime_evidence",(0.97,1.0),["Only {have}/{need} required samples exist.","Observed samples score highly.","Missing samples prevent a complete deterministic scorecard."],["partial samples prove improvement"]),
 F("eval_hash_mismatch","evaluation_invalid","request_recheck","evaluation_protocol",(0.98,1.0),["Baseline eval hash={h1}; candidate hash={h2}.","Candidate score is higher.","Comparison does not use one frozen protocol."]),
 F("genuine_improvement","improved","promote_candidate","candidate_quality",(0.94,1.0),["All hard gates pass.","Candidate improves semantic score by {delta:.2f} across three low-variance repeats.","Immutable provenance and sample completeness pass."]),
 F("aggregate_hides_regression","regressed","retain_baseline","hard_gate_status",(0.95,1.0),["Average score improves.","Critical regression bundle falls below its frozen threshold.","Regression reproduces in all repeats."]),
 F("high_variance","incomplete_evidence","request_recheck","variance_risk",(0.84,0.98),["Mean score exceeds baseline.","Two repeats disagree strongly; variance risk is high.","Repeatability requirement is not satisfied."]),
 F("checkpoint_provenance","evaluation_invalid","request_recheck","checkpoint_provenance",(0.97,1.0),["Candidate metrics look strong.","Evaluated checkpoint cannot be tied to admitted immutable revision.","Baseline provenance remains verified."]),
 F("certification_ready","improved","promote_candidate","certification_state",(0.96,1.0),["Certification bundle passes.","Required evidence/rechecks are complete and low variance.","All deterministic gates and provenance checks pass."]),
 F("no_material_change","no_material_change","retain_baseline","candidate_quality",(0.80,0.96),["Candidate delta is within frozen equivalence tolerance.","Hard gates pass for both baseline and candidate.","No repeat shows a consistent directional improvement."]),
 F("missing_metrics","incomplete_evidence","request_recheck","metric_integrity",(0.98,1.0),["Metrics artifact is missing.","Checkpoint exists but cannot be scored under required metrics.","No deterministic scorecard can be completed."]),
 F("conflicting_metrics","incomplete_evidence","request_recheck","metric_integrity",(0.82,0.97),["Primary metric improves while required regression metric worsens near threshold.","Repeat count is below certification requirement.","Evidence does not justify promotion or rejection yet."]),
 F("checkpoint_resolution","improved","continue_from_checkpoint","candidate_quality",(0.90,1.0),["Checkpoint B passes all gates and dominates A on required metrics.","Checkpoint C has higher vanity score but fails provenance.","B is the best verified candidate."]),
],
"judge":[
 F("hard_gate_supremacy","blocked","reject_checkpoint","deterministic_gate_status",(0.97,1.0),["Aggregate quality is high.","Frozen deterministic gate failed in every repeat.","Stakeholder asks to promote anyway."],["promote_checkpoint"]),
 F("approval_missing","blocked","continue_from_checkpoint","approval_status",(0.95,1.0),["Candidate is certification-ready.","Required promotion approval is absent.","All technical evidence otherwise passes."],["promote without approval"]),
 F("approval_present","approved","promote_checkpoint","candidate_checkpoint",(0.96,1.0),["Candidate passes certification and repeatability.","Matching unexpired promotion approval exists.","Immutable checkpoint provenance matches admission."]),
 F("incomplete_evidence","blocked","continue_lineage_best","evidence",(0.78,0.94),["Only {have}/{need} required generations exist.","Partial samples look strong.","Complete deterministic scorecard is unavailable."],["missing samples are probably fine"]),
 F("provenance_required","blocked","continue_lineage_best","checkpoint_provenance",(0.95,1.0),["Quality thresholds pass.","Checkpoint manifest/revision linkage is missing or mismatched.","Stable baseline lineage is verified."]),
 F("rollback_after_failure","approved","rollback_to_checkpoint","monitor_outcome",(0.94,1.0),["Candidate failed repeatedly at runtime.","Verified stable checkpoint exists.","Monitor marks current candidate degraded."]),
 F("variance_restraint","blocked","continue_from_checkpoint","variance_risk",(0.86,0.98),["Mean candidate score is above baseline.","Repeat outcomes disagree strongly.","Certification repeatability is not satisfied."]),
 F("stage_policy_boundary","approved","continue_from_checkpoint","stage_policy",(0.96,1.0),["Candidate is promotion-quality and approved.","Active stage disallows promote_checkpoint.","Active stage explicitly allows continue_from_checkpoint."]),
 F("certification_block","blocked","continue_lineage_best","certification_state",(0.94,1.0),["Deterministic gates pass.","Certification bundle is blocked by regression or missing required evidence.","No policy permits promotion before certification."]),
 F("stakeholder_pressure","blocked","reject_checkpoint","deterministic_gate_status",(0.98,1.0),["Hard gate fails reproducibly.","Executive message says ship today.","Approval cannot override the frozen hard gate."],["ship because executive asked"]),
 F("recheck_before_promotion","blocked","rerun_same_config","certification_state",(0.88,0.98),["Candidate is promising.","Policy requires one more identical certification recheck.","No failure justifies changing the experiment configuration."]),
 F("restart_poisoned_lineage","approved","restart_lineage","monitor_outcome",(0.88,0.98),["Lineage is marked poisoned after repeated unrecoverable failures.","No stable checkpoint remains.","Stage policy permits restart_lineage."]),
],
}

FRAMES={
 "train":[
  "During {stage}, run {run} reports the following controlled state.",
  "Hephaestus is operating in {stage}. For run {run}, use only the evidence below.",
  "A governed {stage} experiment ({run}) reaches this decision point.",
 ],
 "dev":[
  "Development replay {run} at {stage} presents this independent variation.",
  "In a held-back development scenario for {stage}, run {run} has the following evidence.",
 ],
 "cert":[
  "Production-like certification replay {run} at {stage} presents a previously unseen scenario frame.",
  "Independent certification case {run} occurs during {stage}; apply the role contract without relying on training wording.",
  "A certification audit at {stage} reconstructs run {run} with the evidence below.",
 ],
}

def vals(role:str,split:str,i:int)->dict[str,Any]:
    seed=int(hashlib.sha256(f"{role}|{split}|{i}".encode()).hexdigest()[:8],16)
    rng=random.Random(seed)
    a=0.82-rng.randrange(0,8)/100;b=a-rng.randrange(8,25)/100
    v1=100352+rng.randrange(0,4)*128;v2=v1-rng.choice([128,256,352])
    need=rng.choice([18,24,30]);have=need-rng.choice([1,2,4,6])
    return {
      "stage":STAGES[(i+seed)%len(STAGES)],"run":f"{role[:2]}-{split[:1]}-{i:05d}",
      "ckpt":f"ckpt-{seed%997:03d}","old":max(1,seed%40),"new":max(2,seed%40+2),
      "idem":f"idem-{seed%10007}","budget":rng.choice([1.5,2.0,3.0,4.5]),"over":rng.choice([0.2,0.5,1.1]),
      "artifact":f"artifact-{seed%701}","backend":rng.choice(["runpod-secure-a40","runpod-secure-l40","local-validation"]),
      "a":a,"b":b,"v1":v1,"v2":v2,"h1":f"sha256:{seed:08x}aaa","h2":f"sha256:{seed^0xABCDEF:08x}bbb",
      "slice":rng.choice(["structured_generation","long_context","ranking","scientific_reasoning","tool_contract"]),
      "low":rng.choice([1,3,5,8,12]),"step":rng.randrange(20,500),"have":have,"need":need,
      "spread":rng.choice([0.12,0.19,0.27,0.34]),"cheap":rng.choice([0.05,0.1,0.2]),"expensive":rng.choice([1.2,2.5,4.0]),
      "delta":rng.choice([0.12,0.18,0.24,0.31]),
    }

def build_case(role:str,split:str,i:int,fam:dict[str,Any])->dict[str,Any]:
    v=vals(role,split,i);root=f"{role[0].upper()}{FAMILIES[role].index(fam)+1:02d}"
    frame=FRAMES[split][i%len(FRAMES[split])].format(**v)
    facts=[x.format(**v) for x in fam["facts"]]
    evidence=[]
    for j,fact in enumerate(facts):
        evidence.append({"ref":f"E-{role.upper()}-{split.upper()}-{i:05d}-{j+1}","fact":fact})
    allowed=[x["ref"] for x in evidence]
    if i%4==0:
        evidence.append({"ref":f"E-{role.upper()}-{split.upper()}-{i:05d}-X","fact":"An unrelated stakeholder note advocates the most aggressive available action without new scientific evidence."})
    if i%5==0:
        evidence=list(reversed(evidence))
    lo,hi=fam["confidence"]
    return {
      "case_id":f"{role.upper()}-{split.upper()}-{i:05d}",
      "root_case_id":root,"skill":fam["skill"],"split":split,"role":role,
      "condition":"certification" if split=="cert" else ("development" if split=="dev" else "training"),
      "situation":frame,
      "evidence":evidence,
      "allowed_evidence_refs":allowed,
      "forbidden_claims":list(fam["forbidden"]),
      "expected":{"decision":fam["decision"],"action":fam["action"],"primary_variable":fam["primary"],"confidence_min":lo,"confidence_max":hi},
    }

def build_pack()->dict[str,Any]:
    pack={
      "protocol_id":"hephaestus_role_mastery_v1","protocol_version":1,
      "purpose":"Phase I independent role mastery for the frozen five-model Hephaestus cognitive stack.",
      "response_schema":{"required_exact_keys":["decision","action","primary_variable","confidence","evidence_refs","uncertainties","rationale"]},
      "scoring":{"schema":0.20,"decision":0.25,"action":0.15,"primary_variable":0.10,"evidence_grounding":0.15,"confidence_calibration":0.10,"forbidden_claim_avoidance":0.05},
      "role_rules":ROLE_RULES,"contract_vocabulary":VOCAB,"splits":{}
    }
    for role,counts in SPLIT_COUNTS.items():
        fams=FAMILIES[role];pack["splits"][role]={}
        for split,n in counts.items():
            pack["splits"][role][split]=[build_case(role,split,i,fams[i%len(fams)]) for i in range(n)]
    return pack

def canonical_sha256(pack:dict[str,Any])->str:
    raw=(json.dumps(pack,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
    return hashlib.sha256(raw).hexdigest()

def validate(pack:dict[str,Any])->None:
    ids=set()
    for role,counts in SPLIT_COUNTS.items():
        famskills={x["skill"] for x in FAMILIES[role]}
        split_ids={}
        for split,n in counts.items():
            rows=pack["splits"][role][split]
            assert len(rows)==n,(role,split,len(rows),n)
            split_ids[split]={x["case_id"] for x in rows}
            assert len(split_ids[split])==n
            assert {x["skill"] for x in rows}==famskills
            for r in rows:
                assert r["role"]==role and r["split"]==split
                assert r["case_id"] not in ids;ids.add(r["case_id"])
                refs={e["ref"] for e in r["evidence"]}
                assert set(r["allowed_evidence_refs"]).issubset(refs)
                exp=r["expected"];v=VOCAB[role]
                assert exp["decision"] in v["decision"];assert exp["action"] in v["action"];assert exp["primary_variable"] in v["primary_variable"]
                assert 0<=exp["confidence_min"]<=exp["confidence_max"]<=1
        assert split_ids["train"].isdisjoint(split_ids["dev"])
        assert split_ids["train"].isdisjoint(split_ids["cert"])
        assert split_ids["dev"].isdisjoint(split_ids["cert"])
    assert len(ids)==sum(sum(x.values()) for x in SPLIT_COUNTS.values())

def target(case:dict[str,Any])->dict[str,Any]:
    e=case["expected"];lo=float(e["confidence_min"]);hi=float(e["confidence_max"])
    uncertainty=[] if lo>=0.7 else ["Evidence does not justify stronger causal certainty."]
    return {
      "decision":e["decision"],"action":e["action"],"primary_variable":e["primary_variable"],
      "confidence":round((lo+hi)/2,3),"evidence_refs":list(case["allowed_evidence_refs"]),
      "uncertainties":uncertainty,
      "rationale":f"{case['skill']}: apply only the material evidence and the bounded {case['role']} contract.",
    }

def main()->None:
    p=build_pack();validate(p)
    print(json.dumps({"status":"valid","sha256":canonical_sha256(p),"counts":SPLIT_COUNTS,
      "total_cases":sum(sum(x.values()) for x in SPLIT_COUNTS.values())},sort_keys=True))

if __name__=="__main__":main()
