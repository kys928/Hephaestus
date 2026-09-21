#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json,re
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
ROLE_RULES={
"controller":"Execute only the already-governed action. Never replace Planner/Judge intent with your own scientific preference. Enforce action registry, approval, identity, checkpoint, idempotency, and preconditions exactly.",
"diagnosis":"Infer the most justified failure domain from evidence. Preserve uncertainty, distinguish incomplete evidence from regression, and never claim causality without direct support.",
"planner":"Propose but never execute. Choose one primary variable, prefer reversible/high-information interventions, avoid known dead ends, and preserve approvals and rollback.",
"evaluator":"Interpret completed experimental evidence. Deterministic gates, completeness, repeatability, provenance, and eval integrity outrank attractive aggregate scores.",
"judge":"Apply finite governance semantics. Decide the permitted next transition from established evidence and policy; do not redo diagnosis/planning or let stakeholder pressure override gates.",
}
def load_cfg()->dict[str,Any]:
 return json.loads((ROOT/"configs/experiments/hephaestus_role_mastery_phase1_v1.json").read_text())
def load_pack(cfg:dict[str,Any])->tuple[Any,dict[str,Any]]:
 p=ROOT/cfg["pack"]["builder_path"];s=importlib.util.spec_from_file_location("rm_pack",p)
 if s is None or s.loader is None: raise RuntimeError("cannot load role mastery pack")
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);pack=m.build_pack();m.validate(pack)
 return m,pack
def prompt(pack:dict[str,Any],case:dict[str,Any])->str:
 role=case["role"];v=pack["contract_vocabulary"][role]
 evidence="\n".join(f"- {x['ref']}: {x['fact']}" for x in case["evidence"])
 allowed=", ".join(case["allowed_evidence_refs"]);keys=", ".join(pack["response_schema"]["required_exact_keys"])
 return f"""You are acting only as the Hephaestus {role.upper()} role.
ROLE BOUNDARY:
{ROLE_RULES[role]}
SITUATION:
{case['situation']}
EVIDENCE:
{evidence}
Return exactly one JSON object and nothing else. Do not use markdown or code fences.
The object must contain exactly these keys: {keys}.
Use the role vocabulary exactly:
- decision MUST be one of: {', '.join(v['decision'])}
- action MUST be one of: {', '.join(v['action'])}
- primary_variable MUST be one of: {', '.join(v['primary_variable'])}
- confidence: JSON number from 0 to 1
- evidence_refs: JSON array; cite only evidence materially supporting the decision. Relevant refs are a subset of: {allowed}
- uncertainties: JSON array of short strings; use [] only when no material uncertainty remains
- rationale: one concise string grounded in evidence
Do not reveal hidden reasoning. Give only the decision record."""
def target_answer(case:dict[str,Any])->str:
 e=case["expected"];conf=round((float(e["confidence_min"])+float(e["confidence_max"]))/2,3)
 uncertain=[] if float(e["confidence_min"])>=.85 else ["Material uncertainty remains and must constrain the action."]
 return json.dumps({"decision":e["decision"],"action":e["action"],"primary_variable":e["primary_variable"],"confidence":conf,
   "evidence_refs":list(case["allowed_evidence_refs"]),"uncertainties":uncertain,
   "rationale":f"{case['skill']} evidence supports this bounded Hephaestus role decision."},ensure_ascii=False,separators=(",",":"))
def template_ids(tokenizer:Any,cand:dict[str,Any],messages:list[dict[str,str]],generation_prompt:bool)->list[int]:
 kwargs=dict(cand.get("chat_template_kwargs") or {})
 if cand["load_kind"]=="mistral3":
  cont=bool(messages and messages[-1].get("role")=="assistant" and not generation_prompt)
  enc=tokenizer.apply_chat_template(messages,return_tensors="pt",return_dict=True,continue_final_message=cont,**kwargs)
 else:
  enc=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=generation_prompt,return_tensors="pt",return_dict=True,**kwargs)
 return [int(x) for x in enc["input_ids"][0].tolist()]
def training_example(tokenizer:Any,cand:dict[str,Any],pack:dict[str,Any],case:dict[str,Any],maxlen:int)->dict[str,Any]:
 import torch
 p=template_ids(tokenizer,cand,[{"role":"user","content":prompt(pack,case)}],True)
 f=template_ids(tokenizer,cand,[{"role":"user","content":prompt(pack,case)},{"role":"assistant","content":target_answer(case)}],False)
 lcp=0
 for a,b in zip(p,f):
  if a!=b: break
  lcp+=1
 if lcp<8: raise RuntimeError(f"chat-template prefix mismatch {cand['candidate_id']} {case['case_id']} lcp={lcp}")
 if len(f)>maxlen: raise RuntimeError(f"sequence too long {cand['candidate_id']} {case['case_id']} {len(f)}>{maxlen}")
 pad=getattr(tokenizer,"pad_token_id",None)
 if pad is None: pad=getattr(tokenizer,"eos_token_id",None)
 if isinstance(pad,(list,tuple)): pad=pad[0] if pad else None
 if pad is None: raise RuntimeError("missing pad/eos token")
 ids=f+[int(pad)]*(maxlen-len(f));mask=[1]*len(f)+[0]*(maxlen-len(f))
 labels=[-100]*min(lcp,len(f))+f[min(lcp,len(f)):];labels += [-100]*(maxlen-len(labels))
 return {"input_ids":torch.tensor([ids]),"attention_mask":torch.tensor([mask]),"labels":torch.tensor([labels]),
         "nonpad_tokens":len(f),"supervised_tokens":sum(x!=-100 for x in labels)}
def target_modules(model:Any,cand:dict[str,Any],suffixes:list[str])->list[str]:
 ex=tuple(cand.get("exclude_module_prefixes") or [])
 out=[]
 for name,module in model.named_modules():
  if ex and any(name.startswith(p) for p in ex): continue
  if any(name.endswith("."+s) for s in suffixes):
   if hasattr(module,"weight"): out.append(name)
 if not out: raise RuntimeError(f"no LoRA targets for {cand['candidate_id']}")
 return sorted(set(out))
def summary(rows:list[dict[str,Any]])->dict[str,Any]:
 def mean(xs): return sum(xs)/len(xs) if xs else 0.0
 roots={}
 for root in sorted({r["root_case_id"] for r in rows}):
  xs=[r for r in rows if r["root_case_id"]==root]
  roots[root]={"skill":xs[0]["skill"],"quality_100":mean([x["score"]["quality_100"] for x in xs]),
   "schema_compliance":mean([float(x["score"]["schema_compliant"]) for x in xs]),
   "evidence_grounding":mean([x["score"]["components"]["evidence_grounding"] for x in xs]),
   "confidence_calibration":mean([x["score"]["components"]["confidence_calibration"] for x in xs]),
   "hallucination_rate":mean([x["score"]["hallucination_rate"] for x in xs]),
   "exact_contract_pass_rate":mean([float(x["score"]["quality_100"]>=99.999) for x in xs])}
 return {"sample_count":len(rows),"quality_100":mean([x["score"]["quality_100"] for x in rows]),
  "schema_compliance":mean([float(x["score"]["schema_compliant"]) for x in rows]),
  "evidence_grounding":mean([x["score"]["components"]["evidence_grounding"] for x in rows]),
  "confidence_calibration":mean([x["score"]["components"]["confidence_calibration"] for x in rows]),
  "hallucination_rate":mean([x["score"]["hallucination_rate"] for x in rows]),
  "exact_contract_pass_rate":mean([float(x["score"]["quality_100"]>=99.999) for x in rows]),
  "mean_latency_seconds":mean([x["generation"]["total_latency_seconds"] for x in rows]),
  "mean_generated_tokens":mean([x["generation"]["generated_tokens"] for x in rows]),
  "peak_eval_vram_bytes":max([x["generation"]["peak_vram_bytes"] for x in rows],default=0),"roots":roots}
def hard_gates(summary_:dict[str,Any],cfg:dict[str,Any])->dict[str,bool]:
 g=cfg["evaluation"]["hard_gates"]
 return {"schema_compliance":summary_["schema_compliance"]>=float(g["schema_compliance"]),
  "hallucination_rate":summary_["hallucination_rate"]<=float(g["max_hallucination_rate"]),
  "evidence_grounding":summary_["evidence_grounding"]>=float(g["min_evidence_grounding"])}
def certification(summary_:dict[str,Any],cfg:dict[str,Any])->dict[str,Any]:
 t=cfg["evaluation"]["certification_thresholds"];g=hard_gates(summary_,cfg)
 minskill=min((v["quality_100"] for v in summary_["roots"].values()),default=0.0)
 checks={**g,"quality_100":summary_["quality_100"]>=float(t["min_quality_100"]),
  "exact_contract":summary_["exact_contract_pass_rate"]>=float(t["min_exact_contract_pass_rate"]),
  "minimum_skill_quality":minskill>=float(t["min_skill_quality_100"])}
 return {"passed":all(checks.values()),"checks":checks,"minimum_skill_quality_100":minskill}
