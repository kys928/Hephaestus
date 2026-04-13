from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.config_loader import ConfigError, load_named_config
from hephaestus.schemas.launch_config import LaunchConfig
from hephaestus.schemas.training_plan import TrainingPlan


@dataclass(slots=True)
class TrainingEngineerRole:
    name: str = "training_engineer"
    config_dir: Path = Path("configs")

    def run(
        self,
        run_id: str,
        stage_name: str,
        artifact_root: str,
        data_contract: dict[str, object],
        backend_name: str,
        dry_run: bool,
        recipe_template_name: str | None = None,
    ) -> tuple[TrainingPlan, LaunchConfig]:
        stage_profile = load_named_config(self.config_dir, "stage_profiles", stage_name)
        template_name = recipe_template_name or str(stage_profile.get("recipe_template", ""))
        if not template_name:
            raise ConfigError(f"stage profile '{stage_name}' missing recipe_template")

        template = load_named_config(self.config_dir, "recipe_templates", template_name)
        backend_cfg = load_named_config(self.config_dir, "backends", backend_name)
        parameters = dict(template.get("launch_parameters", {}))
        if not isinstance(parameters, dict):
            raise ConfigError(f"recipe template '{template_name}' launch_parameters must be an object")

        strict_mode = backend_name == "ardor"
        init_mode = self._required_value(
            stage_profile=stage_profile,
            template=template,
            key="init_mode",
            backend_name=backend_name,
            strict_mode=strict_mode,
            default="from_scratch",
        )
        checkpoint_mode = self._required_value(
            stage_profile=stage_profile,
            template=template,
            key="checkpoint_mode",
            backend_name=backend_name,
            strict_mode=strict_mode,
            default="save_final",
        )
        backend_profile = self._required_value(
            stage_profile=backend_cfg,
            template={},
            key="profile",
            backend_name=backend_name,
            strict_mode=strict_mode,
            default=backend_name,
        )
        parameters.update(
            {
                "processed_dataset_ref": str(data_contract["processed_dataset_ref"]),
                "init_mode": init_mode,
                "checkpoint_mode": checkpoint_mode,
                "backend_profile": backend_profile,
            }
        )

        plan = TrainingPlan(
            training_plan_id=f"train-plan-{run_id}",
            run_id=run_id,
            stage_name=stage_name,
            recipe_template=template_name,
            max_steps=int(template["max_steps"]),
            eval_every_steps=int(template["eval_every_steps"]),
            checkpoint_every_steps=int(template["checkpoint_every_steps"]),
            tags=[str(tag) for tag in template.get("tags", ["stage4", stage_name])],
        )
        launch = LaunchConfig(
            launch_id=f"launch-{run_id}",
            run_id=run_id,
            backend=backend_name,
            dry_run=dry_run,
            artifact_root=artifact_root,
            parameters={str(key): str(value) for key, value in parameters.items()},
        )
        return plan, launch

    def _required_value(
        self,
        *,
        stage_profile: dict[str, object],
        template: dict[str, object],
        key: str,
        backend_name: str,
        strict_mode: bool,
        default: str,
    ) -> str:
        if key in stage_profile:
            return str(stage_profile[key])
        if key in template:
            return str(template[key])
        if strict_mode:
            raise ConfigError(f"backend '{backend_name}' requires explicit '{key}' in config")
        return default
