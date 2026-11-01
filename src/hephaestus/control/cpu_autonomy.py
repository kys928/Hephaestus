"""Compose genuine CPU phase owners into the existing production loop/driver."""
from hephaestus.control.staged_state import StagedAutonomousServices,StagedAutonomousDependencies
from hephaestus.production.staged_composition import staged_binding_inventory
from hephaestus.production.staged_driver import StagedActionIntentService
from hephaestus.roles.cpu_autonomy_planning import CpuAutonomyEntryRole,CpuAutonomyPlannerRole
from hephaestus.roles.cpu_autonomy_data import CpuAutonomyDataRole,CpuAutonomyPreprocessorRole
from hephaestus.roles.cpu_autonomy_execution import CpuAutonomyTrainingRole,CpuAutonomyMonitorRole
from hephaestus.roles.cpu_autonomy_evaluation import CpuAutonomyEvaluatorRole,CpuAutonomyJudgeRole


class UnavailableExternalApprovalService:
    def decision_for(self,request):
        # Local finite actions need no new approval. Any unexpected requirement
        # remains pending; this service cannot manufacture an approval receipt.
        return None


def compose_cpu_autonomy(context):
    c=context
    entry=CpuAutonomyEntryRole(c); planner=CpuAutonomyPlannerRole(c)
    data=CpuAutonomyDataRole(c); preprocessor=CpuAutonomyPreprocessorRole(c)
    training=CpuAutonomyTrainingRole(c); monitor=CpuAutonomyMonitorRole(c)
    evaluator=CpuAutonomyEvaluatorRole(c); judge=CpuAutonomyJudgeRole(c)
    services=StagedAutonomousServices(judge_entry=entry,evidence_collector=entry,diagnosis=entry,
        plan_readiness=entry,planner=planner,dataset_discovery=data,dataset_selection=data,dataset_acquisition=data,
        data_preprocessor=preprocessor,model_discovery=training,model_selection=training,runtime_monitor=monitor,
        recovery=monitor,generation=evaluator,evaluator=evaluator,judge_exit=judge,action_executor=StagedActionIntentService())
    services=c.runtime.with_training_services(services,execute_enabled=True)
    deps=StagedAutonomousDependencies(services,c.runtime.state_repository,artifact_store=c.runtime.artifact_store,
                                    approval_service=UnavailableExternalApprovalService())
    staged_binding_inventory(deps)
    return deps
