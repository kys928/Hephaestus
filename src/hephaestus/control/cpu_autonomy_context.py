"""Shared IO for separate bounded CPU phase owners; no role decisions here."""
from dataclasses import dataclass
from pathlib import Path
import json

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.schemas.prospective_model_binding import ProspectiveModelBinding
from hephaestus.schemas.prospective_study import ProspectiveDesign
from hephaestus.schemas.experiment_contract import ExperimentProposal
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.checkpoint_identity_store import CheckpointIdentityStore
from hephaestus.production.staged_records import verified_workflow_payload
from hephaestus.evaluation.prospective_pipeline import file_digest, digest
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest


@dataclass(slots=True)
class CpuAutonomyReadbackRuntime:
    settings: object
    state_repository: object
    artifact_store: object
    loop_state: object
    planner: object


def readback_runtime(root):
    from types import SimpleNamespace
    from hephaestus.storage.sqlite import SQLiteStateRepository
    from hephaestus.storage.filesystem import FileSystemArtifactStore
    from hephaestus.production.state import ProductionLoopStateStore
    from hephaestus.state.experiment_memory import ExperimentMemoryQuery
    repository=SQLiteStateRepository(root/"state"/"production.sqlite3",read_only=True)
    repository.assert_integrity()
    artifacts=FileSystemArtifactStore(root/"artifacts")
    return CpuAutonomyReadbackRuntime(SimpleNamespace(state_root=root/"state"),repository,artifacts,
        ProductionLoopStateStore(repository),SimpleNamespace(memory_query=ExperimentMemoryQuery(root/"state",repository,artifacts)))


@dataclass
class CpuAutonomyContext:
    runtime: object
    state: object
    plan: object
    manifest: dict
    manifest_ref: str
    policy: str
    seed: int
    cycle: int

    @property
    def design(self):
        return ProspectiveDesign(purpose="rehearsal",cases_per_family=4,max_new_tokens=16)

    def check(self,req):
        if (req.run_id != self.state.metadata["staged_run_id"] or req.lineage_id != self.state.lineage_id
                or req.stage_name != "bounded_cpu_research" or self.seed not in self.plan.seeds
                or self.policy not in self.plan.policies):
            raise PermissionError("CPU operation differs from frozen program scope")
        if self.runtime.artifact_store.verify(self.manifest_ref) is not True:
            raise PermissionError("CPU manifest missing or corrupted")
        constraints=self.manifest["training_constraints"]
        if (digest(self.manifest)!=self.manifest_ref or self.manifest["plan"]!=json.loads(json.dumps(self.plan.to_dict()))
                or constraints["max_steps"]!=4 or constraints["device"]!="cpu" or constraints["dtype"]!="float32"
                or constraints["parameter_count"]!=37472 or constraints["local_files_only"] is not True):
            raise PermissionError("CPU manifest or actual resource scope differs from the frozen plan")
        if directory_content_identity(self.manifest["model_path"]) != self.manifest["model_identity"]:
            raise PermissionError("CPU base model changed")
        for path,expected in {**self.manifest["input_files"],**self.manifest.get("method_files",{})}.items():
            if file_digest(path)!=expected:
                raise PermissionError("CPU source/preprocessing bytes changed")

    def source(self,req,substep,kind):
        return verified_workflow_payload(self.runtime.state_repository,self.runtime.artifact_store,req,substep=substep,kind=kind)

    def binding(self,checkpoint=None,run_id=None):
        m=self.manifest
        if not checkpoint:
            return ProspectiveModelBinding("baseline",m["model_path"],m["model_identity"],m["model_path"],m["model_identity"])
        valid,identity=validate_checkpoint_manifest(checkpoint)
        if not valid:
            raise PermissionError("CPU selected checkpoint invalid")
        CheckpointIdentityStore(self.runtime.state_repository).verify(self.state.lineage_id,checkpoint,identity)
        if run_id is None:
            run_id=Path(checkpoint).parent.name
        job=Path(checkpoint).parent/"prepared_job.json"
        return ProspectiveModelBinding(f"seed-{self.seed}",m["model_path"],m["model_identity"],m["model_path"],m["model_identity"],
            self.seed,str(Path(checkpoint)/"model"),directory_content_identity(Path(checkpoint)/"model"),
            checkpoint,identity,str(job),file_digest(job))

    def parent(self):
        lineage=LineageStore(self.runtime.settings.state_root).get_current(self.state.lineage_id)
        return self.binding(lineage.get("best_checkpoint_ref") if lineage else None)

    def proposal(self,req,rate):
        if rate not in self.plan.learning_rate_grid:
            raise PermissionError("CPU candidate rate is outside the admitted finite grid")
        parent=self.parent()
        constraints=dict(self.manifest["training_constraints"])
        constraints["peft_config"]=dict(constraints["peft_config"])
        constraints.update(seed=self.seed,learning_rate=rate,
            fixture_bootstrap_failure=self.cycle==1 and self.state.metadata["staged_training_attempt"]==1)
        if parent.adapter_path:
            constraints["training_mode"]="lora_continuation"
            constraints["peft_config"].update(continuation_kind="adapter_warm_start",
                parent_adapter_ref=parent.adapter_path,parent_adapter_hash=parent.adapter_identity)
        return ExperimentProposal("experiment-"+req.run_id,req.run_id,req.lineage_id,req.stage_name,
            "diagnosis-"+req.run_id,"intervention-"+req.run_id,"learning_rate",
            baseline_ref=digest(parent.identity()),status="ready",training_constraints=constraints,
            controlled_variables={"selected_learning_rate":rate,"seed":self.seed,"policy":self.policy,
                "manifest_ref":self.manifest_ref,"parent_identity":parent.identity(),
                "optimizer_state":"reset_each_cycle_adapter_warm_start"},
            metadata={"intervention_kind":"change_training_recipe","research_scope":"public-tiny-cpu-only"})
