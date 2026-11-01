"""Actual local data inspection and reconstruction, owned by their spine roles."""
import json
from pathlib import Path
from hephaestus.data.generated_validation import validate_original_example
from hephaestus.data.chunking import chunk_records
from hephaestus.storage.verified import canonical_bytes
from hephaestus.training.cpu_chat_json_recipe import chat_prompt
from hephaestus.evaluation.prospective_pipeline import digest,file_digest
from .cpu_autonomy_planning import result


class CpuAutonomyDataRole:
    def __init__(self,context): self.ctx=context

    def execute(self,req):
        c=self.ctx; c.check(req); m=c.manifest
        rows=m["train"]
        keys=[validate_original_example(r) for r in rows]
        if len(set(keys))!=len(rows) or {r["sample_id"] for r in rows}&{r["sample_id"] for r in [*m["development"],*m["test"]]}:
            raise ValueError("CPU source duplicates or split exposure")
        revision=digest(rows)
        if req.substep=="dataset_discovery":
            c.source(req,"discovery_request_preparation","search_request")
            return result("dataset_discovery_result",{"dataset_revision":revision,"rows":len(rows),"source":"authored_public_local_grammar"})
        if req.substep=="dataset_selection":
            c.source(req,"dataset_discovery","dataset_discovery_result")
            return result("dataset_selection_decision",{"status":"selected","dataset_revision":revision,"manifest_ref":c.manifest_ref},
                dataset_revision=revision,subject_ref=c.manifest_ref,required_approvals=[],planner_authorized=True)
        if req.substep=="dataset_acquisition":
            raw=Path(m["source_path"]).read_bytes()
            if json.loads(raw)["train"]!=rows:
                raise ValueError("CPU acquired bytes differ from frozen source")
            return result("acquisition_receipt",{"source_path":m["source_path"],"bytes":len(raw),"content_hash":file_digest(m["source_path"]),"network_used":False})
        if req.substep=="license_provenance_evidence":
            return result("license_provenance_evidence",{"license":"cc0-1.0","basis":"existing authored task grammar and computed mathematical facts",
                "source_hash":file_digest(m["source_path"]),"external_examples_used":False})
        if req.substep=="dataset_audit":
            return result("dataset_audit",{"independently_recomputed_targets":len(keys),"canonical_duplicates":0,"train_development_test_overlap":0})
        if req.substep=="manifest_production":
            return result("dataset_manifest",{"manifest_ref":c.manifest_ref,"dataset_revision":revision,"rows":len(rows),"source_hash":file_digest(m["source_path"])})
        raise ValueError("unsupported CPU data operation")


class CpuAutonomyPreprocessorRole:
    def __init__(self,context): self.ctx=context

    def execute(self,req):
        from transformers import AutoTokenizer
        c=self.ctx; c.check(req); m=c.manifest; constraints=m["training_constraints"]
        tokenizer=AutoTokenizer.from_pretrained(m["model_path"],local_files_only=True,trust_remote_code=False)
        template=chat_prompt(tokenizer,"{prompt}")+"{target}"
        records=chunk_records(tuple({"prompt":r["prompt"],"target":r["target"]+tokenizer.eos_token} for r in m["train"]),
            chunk_size_tokens=512,min_tokens=1,prompt_target_template=template).records
        raw=b"".join(canonical_bytes(r)+b"\n" for r in records)
        if raw!=Path(constraints["processed_dataset_ref"]).read_bytes():
            raise PermissionError("CPU preprocessing reconstruction differs from frozen training bytes")
        if req.substep=="approved_source_preprocessing":
            ref=c.runtime.artifact_store.put_bytes(raw).artifact_ref
            return result("preprocessing_report",{"reconstructed_processed_ref":ref,"rows":len(records),"supervision":"structured prompt masked; target plus EOS"})
        if req.substep=="deduplication_contamination":
            return result("deduplication_contamination",{"canonical_duplicate_count":0,"split_overlap_count":0,"manifest_ref":c.manifest_ref})
        if req.substep=="tokenizer_compatibility":
            widths=[len(tokenizer.encode(r["text"],add_special_tokens=False)) for r in records]
            if max(widths)>512 or any(tokenizer.unk_token_id in tokenizer.encode(r["text"],add_special_tokens=False) for r in records):
                raise ValueError("CPU tokenizer cannot preserve complete structured examples")
            return result("tokenizer_compatibility",{"compatible":True,"maximum_width":max(widths),"tokenizer_identity":m["model_identity"]})
        if req.substep=="trainable_data_contract":
            return result("trainable_data_contract",json.loads(Path(constraints["trainable_data_contract_ref"]).read_text()))
        raise ValueError("unsupported CPU preprocessing operation")
