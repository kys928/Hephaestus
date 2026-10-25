"""Opt-in assembly and auditable coverage of the existing 45-operation spine.

This factory does not create substitute scientific workers, synthesize
approvals, load remote models, or claim execution. All absent live services
fail closed. Only the persisted orchestrator/driver can prove a real run.
"""
from __future__ import annotations

from dataclasses import replace

from hephaestus.control.spine import SPINE_ORDER
from hephaestus.control.staged_autonomous import build_staged_autonomous_orchestrator
from hephaestus.control.staged_state import (
    PHASE_SUBSTEPS, StagedAutonomousDependencies, StagedAutonomousServices,
)
from .staged_driver import StagedActionIntentService


# Owned by GovernedStagedOrchestrator; no external scientific worker is invoked.
_NATIVE_GATES = frozenset({
    "acquisition_approval_gate", "model_approval_gate", "action_approval_gate",
    "action_boundary", "replay_evidence",
})
# These services have no verified staged production adapters on the current
# baseline; require the caller to supply genuine, audited implementations.
_REQUIRED_CORE = (
    "judge_entry", "evidence_collector", "diagnosis", "plan_readiness",
    "planner", "runtime_monitor", "recovery", "judge_exit",
)


def _reject_test_worker(value: object, *, seen: set[int] | None = None) -> None:
    """Reject obvious simulation delegates, including nested fake wrappers.

    Provenance of actual weights is verified later from inference/training
    receipts; a class-name check alone is NEVER evidence of genuine execution.
    """
    if value is None:
        raise ValueError("staged scientific service is missing")
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    cls = type(value)
    name = cls.__name__.lower()
    module = cls.__module__
    if (name.startswith(("fake", "mock", "stub", "simulated", "simulation"))
            or module.startswith(("test_", "tests.", "unittest.mock"))
            or ".test_" in module):
        raise ValueError("test/simulated scientific worker cannot enter production composition")
    if not callable(getattr(value, "execute", None)):
        raise ValueError("staged scientific worker has no execute operation")
    for key in ("delegate", "evaluator", "checkpoint_service", "training"):
        nested = getattr(value, key, None)
        if nested is not None and nested is not value:
            _reject_test_worker(nested, seen=seen)


def staged_binding_inventory(dependencies: StagedAutonomousDependencies) -> dict[str, str]:
    """Verify *all* 45 routes without executing them or writing state.

    Return exact implementation names for review; native gates are explicitly
    distinguished from the other real-service-required operations.
    """
    if dependencies.artifact_store is None:
        raise ValueError("content-addressed artifact custody is required")
    if (dependencies.approval_service is None or
            not callable(getattr(dependencies.approval_service, "decision_for", None))):
        raise ValueError("independent scoped approval provider with decision_for is required")
    orchestrator = build_staged_autonomous_orchestrator(
        workflow_id="coverage-only-not-executed", run_id="coverage-only",
        lineage_id="coverage-only", stage_name="smoke_test",
        dependencies=dependencies,
    )
    bindings: dict[str, str] = {}
    for phase in SPINE_ORDER:
        for substep in PHASE_SUBSTEPS[phase]:
            key = f"{phase.value}.{substep}"
            if substep in _NATIVE_GATES:
                bindings[key] = "GovernedStagedOrchestrator:native_guard"
                continue
            service = orchestrator._service_for(phase, substep)
            _reject_test_worker(service)
            bindings[key] = f"{type(service).__module__}.{type(service).__name__}"
    if len(bindings) != 45 or len(set(bindings)) != 45:
        raise ValueError("45-substep production binding matrix is incomplete")
    return bindings


def compose_verified_staged_dependencies(
    runtime, *, core_services: StagedAutonomousServices | None = None,
    approval_service, reasoner, generation_backend,
    baseline_run_artifact_ref: str,
    approved_recipe_refs: tuple[str, ...],
    approved_baseline_refs: tuple[str, ...],
    enable_local_acquisition: bool = False,
    enable_training_execution: bool = False,
    enable_generation_inference: bool = False,
    eval_pack_name: str = "semantic_behavior_v1",
    diagnosis_request_ref: str | None = None,
) -> StagedAutonomousDependencies:
    """Bind all implemented adapters to one governed, restartable spine.

    Core scientific/role delegates must be separately implemented and
    supplied, not synthesized by this function. Runtime execution remains
    independently opt-in, and admission requires a separate approval service.
    """
    if not approved_recipe_refs or not approved_baseline_refs:
        raise PermissionError("independently pinned recipes and baseline allowlist are required")
    if not isinstance(baseline_run_artifact_ref, str) or not baseline_run_artifact_ref.startswith("sha256:"):
        raise PermissionError("actual immutable baseline handle reference is required")
    if approval_service is None or generation_backend is None or reasoner is None:
        raise ValueError("approval, generation and five-role inference implementations are required")
    if core_services is None:
        if not diagnosis_request_ref:
            raise PermissionError("one immutable diagnosis input must be pinned to compose core services")
        from .staged_core import make_staged_core_services
        core_services = make_staged_core_services(
            runtime, diagnosis_request_ref=diagnosis_request_ref)
    for name in _REQUIRED_CORE:
        _reject_test_worker(getattr(core_services, name))
    # Reject legitimate-looking wrappers around fixture generators. The
    # frozen five-role mastery contract is CUDA-only and must never silently
    # switch to a CPU stand-in. CPU tiny-model proof has a separate scope.
    from hephaestus.llm.certified_runtime import (
        CertifiedRoleReasoner, PinnedTransformersRoleGenerator,
    )
    from hephaestus.generation.backends import TransformersCausalLMGenerationBackend
    if (type(reasoner) is not CertifiedRoleReasoner
            or type(reasoner.generator) is not PinnedTransformersRoleGenerator
            or reasoner.generator.device != "cuda:0"):
        raise PermissionError("production five-role inference requires frozen CUDA model generator")
    if type(generation_backend) is not TransformersCausalLMGenerationBackend:
        raise PermissionError("production generation requires real Transformers inference backend")

    services = runtime.with_discovery_services(
        core_services, require_planner_approval_receipt=True)
    services = runtime.with_local_data_factory(
        services, acquisition_enabled=enable_local_acquisition)
    services = runtime.with_training_services(
        services, execute_enabled=enable_training_execution,
        finalize_verified_selections=True,
        approved_recipe_refs=approved_recipe_refs,
        approved_baseline_refs=approved_baseline_refs,
    )
    services = runtime.with_evaluation_services(
        services, baseline_run_artifact_ref=baseline_run_artifact_ref,
        generation_backend=generation_backend,
        inference_enabled=enable_generation_inference,
        eval_pack_name=eval_pack_name,
    )
    services = runtime.with_role_models(services, reasoner)
    services = replace(services, action_executor=StagedActionIntentService())
    dependencies = StagedAutonomousDependencies(
        services=services, state_repository=runtime.state_repository,
        artifact_store=runtime.artifact_store,
        approval_service=approval_service,
    )
    staged_binding_inventory(dependencies)
    return dependencies
