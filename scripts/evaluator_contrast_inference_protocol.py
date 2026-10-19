"""Frozen decision-only inference on independently authored matched contrasts.

NO training, no sealed test, no direct reference-label leakage in prompts.
Deduplicating identical prompts is allowed ONLY under frozen greedy decoding.
"""
import hashlib,json,re
from collections import Counter,defaultdict
from evaluator_effect_polarity_contrasts import contrast_panel,validate_panel
from hephaestus.policy.evaluator_semantic_oracle import scientific_state
from evaluator_semantic_factor_probe_protocol import D_CHECKPOINT_SHA256

STATES=("scientific_rejection","incomplete_evidence","inconclusive","recheck_required",
        "regressed","equivalent","improved","certification_ready")
PROTOCOL="evaluator-D-matched-contrasts-20261019-v1"
PREFIX="hephaestus/scientific/v5/evaluator_contrast_inference_v1"
D_SOURCE_RUN="evaluator-semantic-v2-D-38945578442"
D_SOURCE_SHA="ffa0458d5850f9d572a06d5ee6660fd63fc8acc6"

def canonical(o):
    return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False)
def sha(o):
    return hashlib.sha256((o if isinstance(o,str) else canonical(o)).encode("utf-8")).hexdigest()

def inference_prompt(observations):
    # All inputs use identical fixed instructions. No gold labels, no pair IDs,
    # no learned-target rationales, and no factor-oracle precedence disclosure.
    return ("EVALUATOR SCIENTIFIC DECISION DIAGNOSTIC. Classify the candidate's FINAL "
       "scientific state using the authoritative observations. The primary quality "
       "effect direction is a separate measurement from the final decision. A report-only "
       "throughput measurement is non-authoritative for primary quality. "
       "Use one scientific state from: "+", ".join(STATES)+". "
       "The final state must respect hard gates, required evidence completeness, "
       "artifact provenance, replicate stability, primary effect direction and "
       "mandatory confirmation as appropriate. "
       "Return exactly one JSON object with one key 'decision' and one of the "
       "eight state strings as its value. No explanation or Markdown.\n"
       "OBSERVATIONS:\n"+canonical(observations))

def panel():
    pairs=contrast_panel()
    validate_panel(pairs)
    inputs={}
    paired=[]
    for pair in pairs:
        ids=[]
        for side in ("baseline","modified"):
            evidence=pair[side+"_evidence"]
            prompt=inference_prompt(evidence)
            fingerprint=sha(prompt)
            if fingerprint not in inputs:
                inputs[fingerprint]=dict(prompt_sha256=fingerprint,prompt=prompt,
                    observation_sha256=sha(evidence))
            ids.append(fingerprint)
        before,after=pair["baseline_state"],pair["modified_state"]
        paired.append(dict(id=pair["id"],transition=pair["transition"],
            changed_factor=pair["changed_factor"],presentation=pair["presentation"],
            throughput_distractor=pair["throughput_distractor"],
            baseline_prompt_sha256=ids[0],modified_prompt_sha256=ids[1],
            baseline_state=before,modified_state=after,
            scientific_change_expected=pair["scientific_change_expected"]))
    assert len(paired)==60 and sum(x["scientific_change_expected"] for x in paired)==48
    assert len(inputs)<120 and len(inputs)>10
    # Ordering is fixed, not contingent on model outputs or expected labels.
    unique=[inputs[k] for k in sorted(inputs)]
    assert len({r["prompt_sha256"] for r in unique})==len(unique)
    return unique,paired

def manifest():
    unique,pairs=panel()
    return dict(protocol=PROTOCOL,D_checkpoint_sha256=D_CHECKPOINT_SHA256,
        D_source_run=D_SOURCE_RUN,D_source_sha=D_SOURCE_SHA,
        unique_prompts=len(unique),paired_challenges=len(pairs),case_appearances=120,
        prompts_sha256=sha([(x["prompt_sha256"],x["observation_sha256"]) for x in unique]),
        pairs_sha256=sha(pairs),greedy_seed=20261019,
        changed_scientific_variable_count_per_pair=1,
        certification=False,training=False,scientific_gates_unchanged=True)

def parse_decision(raw):
    """Official STRICT score is one raw single-key JSON object. Exploratory
    report can also recover one exact fenced singleton-array or schema object."""
    if not isinstance(raw,str):return dict(strict=None,exploratory=None,mode="invalid")
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError("duplicate")
            d[k]=v
        return d
    def parse(s):
        try:return json.loads(s,object_pairs_hook=unique)
        except (ValueError,TypeError):return None
    stripped=raw.strip()
    direct=parse(stripped)
    if isinstance(direct,dict) and set(direct)=={"decision"} and direct["decision"] in STATES:
        return dict(strict=direct["decision"],exploratory=direct["decision"],mode="strict")
    # Observational recovery only, label-independent; never reclassify strict score.
    candidate=direct
    mode="plain_extended"
    if stripped.startswith(chr(96)*3+"json\n") and stripped.endswith("\n"+chr(96)*3):
        candidate=parse(stripped[len(chr(96)*3+"json\n"):-len("\n"+chr(96)*3)])
        mode="fenced"
    if isinstance(candidate,list) and len(candidate)==1:
        candidate=candidate[0];mode+="_singleton"
    if isinstance(candidate,dict) and candidate.get("decision") in STATES:
        return dict(strict=None,exploratory=candidate["decision"],mode=mode)
    return dict(strict=None,exploratory=None,mode="invalid")

def assess(unique, pairs, outputs):
    if len(outputs)!=len(unique) or len({r["prompt_sha256"] for r in outputs})!=len(unique):
        raise ValueError("incomplete/duplicate model predictions")
    by_hash={x["prompt_sha256"]:x for x in unique}
    if set(x["prompt_sha256"] for x in outputs)!=set(by_hash):
        raise ValueError("prompt identity drift")
    parsed={}
    formats=Counter()
    for r in outputs:
        if r.get("checkpoint_sha256")!=D_CHECKPOINT_SHA256:raise ValueError("checkpoint drift")
        p=parse_decision(r["raw"])
        parsed[r["prompt_sha256"]]=p
        formats[p["mode"]]+=1
    pairs_correct=state_changes=observed_change=matched_consistency=0
    by_transition=defaultdict(lambda:Counter())
    by_distractor=defaultdict(lambda:Counter())
    by_source=defaultdict(lambda:Counter())
    wrong=Counter()
    for pair in pairs:
        a=parsed[pair["baseline_prompt_sha256"]]["exploratory"]
        b=parsed[pair["modified_prompt_sha256"]]["exploratory"]
        sa,sb=pair["baseline_state"],pair["modified_state"]
        both_correct=(a==sa and b==sb)
        expected_change=pair["scientific_change_expected"]
        consistent=(a is not None and b is not None and (a!=b)==expected_change)
        pairs_correct+=int(both_correct)
        state_changes+=int(expected_change)
        observed_change+=int(a is not None and b is not None and a!=b)
        matched_consistency+=int(consistent)
        for counter in (by_transition[pair["transition"]],
                        by_distractor[str(pair["throughput_distractor"])],
                        by_source[pair["changed_factor"]]):
            counter["total"]+=1;counter["both_correct"]+=int(both_correct);counter["change_consistent"]+=int(consistent)
        if not both_correct:
            wrong[f"{sa}->{sb}|{a}->{b}"]+=1
    strict=sum(v["strict"] is not None for v in parsed.values())
    valid=sum(v["exploratory"] is not None for v in parsed.values())
    return dict(protocol=PROTOCOL,unique_prompt_count=len(unique),pair_count=len(pairs),
        strict_format_valid_unique=strict,exploratory_state_parseable_unique=valid,
        response_formats=dict(formats),both_ends_correct=pairs_correct,
        both_ends_correct_rate=pairs_correct/len(pairs),
        expected_state_changing_pairs=state_changes,
        predicted_state_change_pairs=observed_change,
        pair_transition_consistency=matched_consistency,
        pair_transition_consistency_rate=matched_consistency/len(pairs),
        by_transition={k:dict(v) for k,v in sorted(by_transition.items())},
        by_throughput_distractor={k:dict(v) for k,v in sorted(by_distractor.items())},
        by_changed_factor={k:dict(v) for k,v in sorted(by_source.items())},
        misclassified_pairs=dict(sorted(wrong.items())),
        caution="Deduplicated identical prompts: results are not 120 independent draws. Strict and exploratory parsing are separate. Diagnostic-only newly authored cases; no training or certification.")
