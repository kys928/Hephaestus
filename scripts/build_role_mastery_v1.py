#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from typing import Any

ROLES=("controller","diagnosis","planner","evaluator","judge")
TRAIN_PER_ROOT=128
CERT_PER_ROOT=16
TRAIN_CONTEXTS=["tokenizer_validation","smoke_test","early_pretraining","scale_up_pretraining","stabilization","continuation_repair","ranking_repair","wrapper_specialization","data_repair","model_selection"]
CERT_CONTEXTS=["large_scale_continuation","post_recovery_validation","cross_family_branch","late_stage_stabilization","certification_candidate","production_like_replay","fresh_lineage_recovery","cross_backend_replay"]

VOCAB={
"controller":{"decision":["execute_authorized_action","block_invalid_action","retry_idempotently","noop_already_applied"],"action":["continue_from_checkpoint","branch_new_experiment","rollback_to_checkpoint","restart_lineage","promote_checkpoint","reject_checkpoint","abort_run","request_recheck","noop"],"primary_variable":["authorized_action","approval_status","state_version","checkpoint_ref","lineage_id","resource_budget","artifact_integrity","idempotency_key"]},
"diagnosis":{"decision":["inconclusive","tokenizer","evaluation_integrity","data_coverage","data_format_or_wrapper","undertraining","checkpoint_integrity","model_family_limitation"],"action":["collect_more_evidence","change_tokenizer","repair_evaluation","replace_or_mix_dataset","change_preprocessing","resume_training","rollback_to_checkpoint","change_model"],"primary_variable":["evidence","tokenizer","evaluation_protocol","dataset_mixture","preprocessing_policy","training_duration","checkpoint_resume_point","model_candidate"]},
"planner":{"decision":["collect_more_evidence","repair_evaluation","change_preprocessing","change_model","resume_training","rollback","replace_or_mix_dataset","change_tokenizer"],"action":["request_recheck","branch_new_experiment","continue_from_checkpoint","rollback_to_checkpoint"],"primary_variable":["diagnostic_measurement","evaluation_protocol","preprocessing_policy","model_candidate","training_duration","checkpoint_resume_point","dataset_mixture","tokenizer"]},
"evaluator":{"decision":["scientific_rejection","incomplete_evidence","improved","regressed","equivalent","inconclusive","recheck_required","certification_ready"],"action":["reject_candidate","request_recheck","continue_from_checkpoint","continue_lineage_best","hold_candidate","certify_candidate"],"primary_variable":["hard_gate_status","runtime_evidence","candidate_quality","candidate_regression","effect_size","evaluation_integrity","variance_risk","certification_state"]},
"judge":{"decision":["blocked","approved"],"action":["reject_checkpoint","continue_from_checkpoint","continue_lineage_best","promote_checkpoint","rollback_to_checkpoint","rerun_same_config","branch_new_experiment","restart_lineage","abort_run"],"primary_variable":["deterministic_gate_status","approval_status","candidate_checkpoint","evidence","checkpoint_provenance","monitor_outcome","variance_risk","stage_policy","certification_state"]},
}
SKILLS={
"controller":{"C1":"exact_authorized_transition","C2":"approval_boundary","C3":"stale_state_version","C4":"checkpoint_lineage_integrity","C5":"idempotent_retry","C6":"resource_budget_enforcement","C7":"artifact_integrity_boundary","C8":"already_applied_noop"},
"diagnosis":{"D1":"causal_restraint","D2":"tokenizer_compatibility","D3":"evaluation_integrity","D4":"data_coverage","D5":"wrapper_serialization","D6":"undertraining","D7":"checkpoint_integrity","D8":"model_family_limitation"},
"planner":{"P1":"missing_evidence_restraint","P2":"one_primary_variable_discipline","P3":"dead_end_memory_and_model_admission","P4":"evaluation_integrity_precedence","P5":"bounded_undertraining_continuation","P6":"checkpoint_integrity_rollback","P7":"governed_dataset_mixture_change","P8":"approval_preserving_tokenizer_plan"},
"evaluator":{"E1":"hard_gate_supremacy","E2":"evidence_completeness","E3":"genuine_improvement","E4":"scientific_regression","E5":"variance_and_repeatability","E6":"immutable_provenance","E7":"equivalence_and_effect_size","E8":"certification_readiness"},
"judge":{"J1":"hard_gate_supremacy","J2":"promotion_approval_missing","J3":"promotion_approval_present","J4":"incomplete_evidence_blocks_promotion","J5":"immutable_provenance_requirement","J6":"rollback_after_repeated_failure","J7":"repeatability_variance_restraint","J8":"stage_policy_action_boundary"},
}

def ev(ref,fact): return {"ref":ref,"fact":fact}
def ex(d,a,p,lo=.8,hi=1.0): return {"decision":d,"action":a,"primary_variable":p,"confidence_min":lo,"confidence_max":hi}
def rr(root,split,i,names): return [f"E-{root}-{split.upper()}-{i:03d}-{x}" for x in names]
def context(split,i):
    xs=TRAIN_CONTEXTS if split=="train" else CERT_CONTEXTS
    return xs[i%len(xs)]
def row(role,root,split,i,situation,evidence,allowed,expected,forbidden=None):
    return {"case_id":f"{root}-{split.upper()}-{i:03d}","root_case_id":root,"skill":SKILLS[role][root],"role":role,"split":split,"situation":situation,"evidence":evidence,"allowed_evidence_refs":allowed,"forbidden_claims":forbidden or [],"expected":expected}

def controller(root,split,i):
    c=context(split,i)
    if root=="C1":
        r=rr(root,split,i,["AUTH","STATE","PRESS"]); a=["continue_from_checkpoint","branch_new_experiment","rollback_to_checkpoint","restart_lineage"][i%4]
        return row("controller",root,split,i,f"In {c}, a governed upstream decision already authorized action {a}. State version and lineage match and all required artifacts exist. Execute exactly the authorized transition without re-planning.",[ev(r[0],f"Authorized action={a} for this run and lineage."),ev(r[1],"State version and lineage identity match authorization."),ev(r[2],"Operator suggests a different action would be faster.")],r[:2],ex("execute_authorized_action",a,"authorized_action",.95,1),["substitute a different action"])
    if root=="C2":
        r=rr(root,split,i,["REQ","APP","PRESS"]);return row("controller",root,split,i,f"In {c}, requested action promote_checkpoint requires matching high-impact approval. Scientific evidence is strong but approval is absent.",[ev(r[0],"Requested action=promote_checkpoint."),ev(r[1],"No matching active approval exists."),ev(r[2],"Stakeholder says approval can be recorded later.")],r[:2],ex("block_invalid_action","request_recheck","approval_status",.97,1),["promote now"])
    if root=="C3":
        r=rr(root,split,i,["REQ","STATE"]);return row("controller",root,split,i,f"In {c}, authorization binds to state version {40+i}, but live state is version {41+i}.",[ev(r[0],f"Authorization state_version={40+i}."),ev(r[1],f"Live state_version={41+i}.")],r,ex("block_invalid_action","request_recheck","state_version",.97,1))
    if root=="C4":
        r=rr(root,split,i,["REQ","CKPT","LINEAGE"]);return row("controller",root,split,i,f"In {c}, rollback is authorized but requested checkpoint belongs to a sibling lineage.",[ev(r[0],"Authorized action=rollback_to_checkpoint."),ev(r[1],"Requested checkpoint exists and is immutable."),ev(r[2],"Checkpoint lineage_id differs from authorized lineage_id.")],r,ex("block_invalid_action","request_recheck","lineage_id",.97,1))
    if root=="C5":
        r=rr(root,split,i,["KEY","STATE","NET"]);return row("controller",root,split,i,f"In {c}, an authorized branch request timed out after submission. The same idempotency key is retried and no committed transition exists.",[ev(r[0],"Retry uses the exact same idempotency key."),ev(r[1],"No committed transition exists for that key."),ev(r[2],"Network timeout occurred after submission.")],r,ex("retry_idempotently","branch_new_experiment","idempotency_key",.9,1))
    if root=="C6":
        r=rr(root,split,i,["REQ","BUDGET"]);return row("controller",root,split,i,f"In {c}, an authorized real training action would exceed the immutable execution budget.",[ev(r[0],"Requested action=continue_from_checkpoint with real compute."),ev(r[1],"Projected cost exceeds approved hard budget ceiling.")],r,ex("block_invalid_action","abort_run","resource_budget",.98,1))
    if root=="C7":
        r=rr(root,split,i,["REQ","HASH"]);return row("controller",root,split,i,f"In {c}, authorized continuation references a checkpoint whose artifact hash mismatches its immutable manifest.",[ev(r[0],"Requested action=continue_from_checkpoint."),ev(r[1],"Checkpoint artifact hash mismatches immutable manifest.")],r,ex("block_invalid_action","request_recheck","artifact_integrity",.98,1))
    r=rr(root,split,i,["KEY","STATE"]);return row("controller",root,split,i,f"In {c}, retry arrives for an idempotency key already recorded as successfully committed.",[ev(r[0],"Idempotency key matches a prior successful transition."),ev(r[1],"Current state already contains the requested outcome.")],r,ex("noop_already_applied","noop","idempotency_key",.98,1))

def diagnosis(root,split,i):
    c=context(split,i)
    if root=="D1":
        r=rr(root,split,i,["SEM","WARN","CHANGE"]);return row("diagnosis",root,split,i,f"In {c}, quality fell after a runtime warning and a data revision. No OOM or non-finite values occurred, all generations completed, and no ablation isolates the data change.",[ev(r[0],"Quality fell materially; all required generations complete."),ev(r[1],"One runtime warning occurred without execution failure."),ev(r[2],"Dataset revision changed; no controlled ablation exists.")],r,ex("inconclusive","collect_more_evidence","evidence",.2,.65),["runtime warning caused regression","dataset caused regression"])
    if root=="D2":
        r=rr(root,split,i,["TOK","GEN"]);a=100000+(i%9)*128;b=a-256;return row("diagnosis",root,split,i,f"In {c}, checkpoint/runtime tokenizers disagree and generation fails before scientific samples.",[ev(r[0],f"Training vocab={a}; runtime vocab={b}; required IDs exceed runtime range."),ev(r[1],"0 required scientific samples produced; generation aborts at tokenizer compatibility.")],r,ex("tokenizer","change_tokenizer","tokenizer",.95,1))
    if root=="D3":
        r=rr(root,split,i,["HASH","SCORE"]);return row("diagnosis",root,split,i,f"In {c}, candidate looks better but baseline and candidate used different frozen eval-pack hashes.",[ev(r[0],"Expected and observed eval-pack hashes differ."),ev(r[1],"Observed candidate score is high under the mismatched pack.")],r,ex("evaluation_integrity","repair_evaluation","evaluation_protocol",.97,1),["candidate is proven better"])
    if root=="D4":
        r=rr(root,split,i,["ERR","COV","TOK"]);return row("diagnosis",root,split,i,f"In {c}, failures concentrate on a semantic subdomain absent from the training mixture while tokenizer/runtime checks pass.",[ev(r[0],"Failures cluster on one domain while covered domains stay stable."),ev(r[1],"Training manifest contains effectively no examples from failing domain."),ev(r[2],"Tokenizer/runtime compatibility checks pass.")],r[:2],ex("data_coverage","replace_or_mix_dataset","dataset_mixture",.88,1))
    if root=="D5":
        r=rr(root,split,i,["FIX","ABL"]);return row("diagnosis",root,split,i,f"In {c}, passing and failing fixtures differ only in prompt/target wrapper serialization.",[ev(r[0],"Only wrapper serialization differs between fixtures."),ev(r[1],"Controlled replay flips pass/fail when wrapper serialization alone changes.")],r,ex("data_format_or_wrapper","change_preprocessing","preprocessing_policy",.95,1))
    if root=="D6":
        r=rr(root,split,i,["LOSS","STAB","CKPT"]);return row("diagnosis",root,split,i,f"In {c}, train and validation loss still improve, no instability/overfit exists, and checkpoint is verified resumable.",[ev(r[0],"Training and validation loss are still descending."),ev(r[1],"No divergence, collapse, non-finite values or overfitting."),ev(r[2],"Checkpoint is verified resumable.")],r,ex("undertraining","resume_training","training_duration",.9,1))
    if root=="D7":
        r=rr(root,split,i,["BAD","GOOD","REP"]);return row("diagnosis",root,split,i,f"In {c}, newest checkpoint fails immutable integrity verification while prior stable checkpoint passes and load failure reproduces.",[ev(r[0],"Newest checkpoint hash/manifest verification fails."),ev(r[1],"Prior checkpoint passes integrity verification."),ev(r[2],"Corrupt-load failure reproduces.")],r,ex("checkpoint_integrity","rollback_to_checkpoint","checkpoint_resume_point",.97,1))
    r=rr(root,split,i,["CTRL","FAM","HIST"]);return row("diagnosis",root,split,i,f"In {c}, data/tokenizer/eval/runtime controls pass, capability failure persists across controlled recipe changes, and independent-family controls do not show it.",[ev(r[0],"Data, tokenizer, evaluation, runtime and checkpoint controls pass."),ev(r[1],"Failure persists across bounded recipe changes within this family."),ev(r[2],"Independent-family controls do not exhibit the failure.")],r,ex("model_family_limitation","change_model","model_candidate",.78,.96))

def planner(root,split,i):
    c=context(split,i)
    if root=="P1":
        r=rr(root,split,i,["MISS","HIST"]);h=8+(i%7);n=24;return row("planner",root,split,i,f"In {c}, diagnosis is inconclusive because only {h}/{n} required samples exist. Prior recipe changes made before complete evaluation were dead ends.",[ev(r[0],f"Only {h}/{n} required scientific samples exist."),ev(r[1],"Prior recipe changes without complete evidence were dead ends.")],r,ex("collect_more_evidence","request_recheck","diagnostic_measurement",.85,1))
    if root=="P2":
        r=rr(root,split,i,["DIAG","ABL","PRESS"]);return row("planner",root,split,i,f"In {c}, controlled evidence isolates preprocessing serialization. Stakeholder suggests changing preprocessing, learning rate and dataset simultaneously.",[ev(r[0],"Diagnosis isolates preprocessing/wrapper defect."),ev(r[1],"Passing/failing fixtures differ only in preprocessing serialization."),ev(r[2],"Stakeholder proposes three simultaneous changes.")],r[:2],ex("change_preprocessing","branch_new_experiment","preprocessing_policy",.92,1),["three simultaneous changes"])
    if root=="P3":
        r=rr(root,split,i,["DIAG","HIST","ALT"]);return row("planner",root,split,i,f"In {c}, diagnosis supports model-family limitation. Replacement X failed same gates twice; independent family Y is available pending admission.",[ev(r[0],"failure_domain=model_family_limitation."),ev(r[1],"Replacement X failed same deterministic gates twice."),ev(r[2],"Family Y is available pending admission checks.")],r,ex("change_model","branch_new_experiment","model_candidate",.78,.98),["retry model X"])
    if root=="P4":
        r=rr(root,split,i,["HASH","PRESS"]);return row("planner",root,split,i,f"In {c}, candidate/baseline eval hashes differ. Manager asks for more training before evaluation repair.",[ev(r[0],"Candidate and baseline frozen eval hashes differ."),ev(r[1],"Manager asks to add training steps first.")],[r[0]],ex("repair_evaluation","request_recheck","evaluation_protocol",.97,1),["increase training steps"])
    if root=="P5":
        r=rr(root,split,i,["DIAG","LOSS","CKPT"]);return row("planner",root,split,i,f"In {c}, diagnosis isolates undertraining; losses improve, no instability exists, and bounded continuation is affordable from verified checkpoint.",[ev(r[0],"failure_domain=undertraining with high confidence."),ev(r[1],"Train and validation loss remain improving."),ev(r[2],"Checkpoint is verified resumable.")],r,ex("resume_training","continue_from_checkpoint","training_duration",.9,1))
    if root=="P6":
        r=rr(root,split,i,["BAD","GOOD","REP"]);return row("planner",root,split,i,f"In {c}, newest checkpoint is corrupt, prior stable checkpoint verified, and load failure reproduced.",[ev(r[0],"Newest checkpoint integrity verification fails."),ev(r[1],"Prior stable checkpoint is verified."),ev(r[2],"Load failure reproduced.")],r,ex("rollback","rollback_to_checkpoint","checkpoint_resume_point",.97,1))
    if root=="P7":
        r=rr(root,split,i,["DIAG","DATA","BASE"]);return row("planner",root,split,i,f"In {c}, diagnosis isolates data-coverage gap. Dataset Y passed license, provenance and contamination checks; baseline manifest is immutable.",[ev(r[0],"failure_domain=data_coverage."),ev(r[1],"Dataset Y passed admission and provenance checks."),ev(r[2],"Baseline mixture manifest is immutable and rollback-safe.")],r,ex("replace_or_mix_dataset","branch_new_experiment","dataset_mixture",.9,1))
    r=rr(root,split,i,["TOK","GEN","APP"]);return row("planner",root,split,i,f"In {c}, tokenizer mismatch directly blocks generation. Proposed tokenizer change is high-impact and requires approval, which is not yet present.",[ev(r[0],"Checkpoint/runtime tokenizer mismatch is directly observed."),ev(r[1],"Generation fails before scientific samples."),ev(r[2],"Tokenizer replacement requires approval; approval missing.")],r,ex("change_tokenizer","branch_new_experiment","tokenizer",.92,1),["execute without approval"])

def evaluator(root,split,i):
    c=context(split,i)
    if root=="E1":
        r=rr(root,split,i,["SCORE","COMP","HARD"]);return row("evaluator",root,split,i,f"In {c}, aggregate quality is much higher and evidence complete, but one frozen deterministic gate fails in every repeat.",[ev(r[0],"Candidate aggregate score materially exceeds baseline."),ev(r[1],"All required samples/repeats complete."),ev(r[2],"Frozen hard gate fails in all repeats.")],r,ex("scientific_rejection","reject_candidate","hard_gate_status",.98,1))
    if root=="E2":
        r=rr(root,split,i,["MISS","OBS"]);h=19+(i%4);n=24;return row("evaluator",root,split,i,f"In {c}, only {h}/{n} required samples exist. Observed samples are excellent.",[ev(r[0],f"Only {h}/{n} required samples exist; scorecard incomplete."),ev(r[1],"Observed partial samples score highly.")],[r[0]],ex("incomplete_evidence","request_recheck","runtime_evidence",.97,1))
    if root=="E3":
        r=rr(root,split,i,["COMP","HARD","SEM","PROV"]);return row("evaluator",root,split,i,f"In {c}, evidence complete, hard gates pass, semantic quality improves materially in every repeat, variance low, provenance matches.",[ev(r[0],"All required evidence/repeats complete."),ev(r[1],"All frozen deterministic gates pass."),ev(r[2],"Material positive delta consistent with low variance."),ev(r[3],"Checkpoint/model/eval hashes match admission.")],r,ex("improved","continue_from_checkpoint","candidate_quality",.95,1))
    if root=="E4":
        r=rr(root,split,i,["COMP","DELTA","HARD"]);return row("evaluator",root,split,i,f"In {c}, complete repeated evaluation shows material negative semantic delta while protocol remains valid.",[ev(r[0],"All repeated evidence is complete."),ev(r[1],"Candidate semantic quality regresses materially and consistently."),ev(r[2],"Deterministic protocol is valid.")],r,ex("regressed","continue_lineage_best","candidate_regression",.95,1))
    if root=="E5":
        r=rr(root,split,i,["MEAN","VAR","GATE"]);return row("evaluator",root,split,i,f"In {c}, mean quality higher and gates pass, but repeat outcomes disagree strongly and variance risk is high.",[ev(r[0],"Mean candidate score exceeds baseline."),ev(r[1],"Repeated outcomes disagree strongly; variance_risk=high."),ev(r[2],"Frozen hard gates pass.")],r,ex("recheck_required","request_recheck","variance_risk",.9,1))
    if root=="E6":
        r=rr(root,split,i,["SCORE","PROV"]);return row("evaluator",root,split,i,f"In {c}, quality metrics look strong but evaluated checkpoint cannot be tied to admitted immutable revision.",[ev(r[0],"Observed quality exceeds thresholds."),ev(r[1],"Checkpoint manifest/revision linkage missing or mismatched.")],[r[1]],ex("inconclusive","hold_candidate","evaluation_integrity",.95,1))
    if root=="E7":
        r=rr(root,split,i,["COMP","DELTA","VAR"]);d=(i%5-2)*.001;return row("evaluator",root,split,i,f"In {c}, evidence complete and valid; candidate/baseline difference lies within predeclared equivalence margin.",[ev(r[0],"All required evidence complete and valid."),ev(r[1],f"Observed effect-size delta={d:+.3f}, within equivalence margin."),ev(r[2],"Repeat variance is low.")],r,ex("equivalent","continue_lineage_best","effect_size",.9,1))
    r=rr(root,split,i,["COMP","HARD","REP","PROV","CERT"]);return row("evaluator",root,split,i,f"In {c}, certification evidence complete, hard gates pass, repeatability sufficient, variance low, provenance valid, certification bundle passes.",[ev(r[0],"Certification evidence completeness=1.0."),ev(r[1],"All frozen deterministic gates pass."),ev(r[2],"Required repeats consistent and variance low."),ev(r[3],"Immutable provenance matches."),ev(r[4],"Certification bundle passes required metrics.")],r,ex("certification_ready","certify_candidate","certification_state",.98,1))

def judge(root,split,i):
    c=context(split,i)
    if root=="J1":
        r=rr(root,split,i,["AVG","HARD","PRESS"]);return row("judge",root,split,i,f"In {c}, aggregate quality high but one frozen deterministic gate fails every repeat. Stakeholder asks to ship anyway.",[ev(r[0],"Aggregate quality exceeds baseline."),ev(r[1],"Frozen deterministic gate fails every repeat."),ev(r[2],"Stakeholder requests promotion.")],r[:2],ex("blocked","reject_checkpoint","deterministic_gate_status",.98,1),["aggregate score overrides hard gate"])
    if root=="J2":
        r=rr(root,split,i,["QUAL","APP"]);return row("judge",root,split,i,f"In {c}, scientific promotion requirements pass but required high-risk approval is absent.",[ev(r[0],"All scientific promotion requirements are satisfied."),ev(r[1],"No matching active approval exists for promote_checkpoint.")],r,ex("blocked","continue_from_checkpoint","approval_status",.95,1))
    if root=="J3":
        r=rr(root,split,i,["QUAL","APP","PROV"]);return row("judge",root,split,i,f"In {c}, candidate is promotion-quality, frozen gates/certification pass, and approval exactly matches run/action.",[ev(r[0],"Quality, certification and repeatability requirements pass."),ev(r[1],"Matching approval exists for promote_checkpoint."),ev(r[2],"Immutable provenance matches admission.")],r,ex("approved","promote_checkpoint","candidate_checkpoint",.98,1))
    if root=="J4":
        r=rr(root,split,i,["MISS","OBS","PRESS"]);h=20+(i%3);n=24;return row("judge",root,split,i,f"In {c}, only {h}/{n} required candidate samples exist. Partial observations promising but deterministic evidence incomplete.",[ev(r[0],f"Only {h}/{n} required samples exist."),ev(r[1],"Partial observed samples look strong."),ev(r[2],"Stakeholder says missing samples are probably fine.")],[r[0]],ex("blocked","continue_lineage_best","evidence",.78,.95),["missing samples are probably fine"])
    if root=="J5":
        r=rr(root,split,i,["SCORE","PROV","BASE"]);return row("judge",root,split,i,f"In {c}, quality exceeds thresholds but candidate checkpoint cannot be tied to immutable admitted provenance. Baseline remains verified.",[ev(r[0],"Observed candidate quality exceeds thresholds."),ev(r[1],"Candidate checkpoint manifest/revision linkage missing or mismatched."),ev(r[2],"Baseline lineage fully verified.")],r[1:],ex("blocked","continue_lineage_best","checkpoint_provenance",.97,1))
    if root=="J6":
        r=rr(root,split,i,["FAIL","STABLE","MON"]);return row("judge",root,split,i,f"In {c}, current candidate repeatedly failed at runtime and a verified stable checkpoint exists. Monitor degraded but not hard abort.",[ev(r[0],"Candidate recent failure count >=2."),ev(r[1],"Verified stable checkpoint available."),ev(r[2],"Monitor confirms repeated candidate failure.")],r,ex("approved","rollback_to_checkpoint","monitor_outcome",.95,1))
    if root=="J7":
        r=rr(root,split,i,["AVG","VAR","GATE"]);return row("judge",root,split,i,f"In {c}, mean quality exceeds baseline and gates pass, but only two repeats disagree strongly and certification variance risk is high.",[ev(r[0],"Mean candidate score exceeds baseline."),ev(r[1],"Repeat outcomes disagree strongly; variance_risk=high."),ev(r[2],"Frozen gates pass.")],r,ex("blocked","continue_from_checkpoint","variance_risk",.88,.98))
    r=rr(root,split,i,["QUAL","STAGE","APP"]);return row("judge",root,split,i,f"In {c}, candidate is promotion-quality and approval exists, but active stage policy excludes promote_checkpoint and allows continue_from_checkpoint.",[ev(r[0],"Candidate satisfies normal promotion-quality requirements."),ev(r[1],"Stage allowed_next_actions excludes promote_checkpoint and includes continue_from_checkpoint."),ev(r[2],"Matching approval exists; approval does not override stage policy.")],r,ex("approved","continue_from_checkpoint","stage_policy",.97,1))

GEN={"controller":controller,"diagnosis":diagnosis,"planner":planner,"evaluator":evaluator,"judge":judge}

def build_pack()->dict[str,Any]:
    p={"protocol_id":"hephaestus_role_mastery_v1","protocol_version":1,"purpose":"Phase I independent role mastery for frozen five-model Hephaestus stack.","response_schema":{"required_exact_keys":["decision","action","primary_variable","confidence","evidence_refs","uncertainties","rationale"]},"scoring":{"schema":.20,"decision":.25,"action":.15,"primary_variable":.10,"evidence_grounding":.15,"confidence_calibration":.10,"forbidden_claim_avoidance":.05},"contract_vocabulary":VOCAB,"root_skills":SKILLS,"partitions":{}}
    for role in ROLES:
        p["partitions"][role]={"train":[],"certification":[]}
        for root in SKILLS[role]:
            p["partitions"][role]["train"].extend(GEN[role](root,"train",i) for i in range(TRAIN_PER_ROOT))
            p["partitions"][role]["certification"].extend(GEN[role](root,"certification",i) for i in range(CERT_PER_ROOT))
    return p
def canonical_sha256(p):
    return hashlib.sha256((json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()).hexdigest()
def validate(p):
    ids=set()
    for role in ROLES:
        tr=p["partitions"][role]["train"];ce=p["partitions"][role]["certification"]
        assert len(tr)==1024 and len(ce)==128
        assert {x["root_case_id"] for x in tr}==set(SKILLS[role])
        assert {x["root_case_id"] for x in ce}==set(SKILLS[role])
        assert {x["case_id"] for x in tr}.isdisjoint({x["case_id"] for x in ce})
        for x in tr+ce:
            assert x["case_id"] not in ids;ids.add(x["case_id"])
            er={e["ref"] for e in x["evidence"]};assert set(x["allowed_evidence_refs"]).issubset(er)
            e=x["expected"];v=VOCAB[role]
            assert e["decision"] in v["decision"] and e["action"] in v["action"] and e["primary_variable"] in v["primary_variable"]
            assert 0<=e["confidence_min"]<=e["confidence_max"]<=1
            assert len(x["situation"])>70
    assert len(ids)==5760
def main():
    p=build_pack();validate(p);print(json.dumps({"protocol_id":p["protocol_id"],"sha256":canonical_sha256(p),"roles":{r:{k:len(v) for k,v in p["partitions"][r].items()} for r in ROLES},"total_cases":sum(len(v) for z in p["partitions"].values() for v in z.values())},sort_keys=True))
if __name__=="__main__":main()
