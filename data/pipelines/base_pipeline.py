"""Simple sequential pipeline runner."""

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from data.pipelines.base_step import BaseStep
from utils.builders.builder import build_from_config


class BasePipeline:
    """Executes a list of pipeline components sequentially."""

    PIPELINE_NAME: ClassVar[str] = "base_pipeline"

    def __init__(
        self,
        step_configs: Sequence[Dict[str, Any]],
        input_root: str | Path,
        output_root: str | Path,
    ) -> None:
        assert isinstance(step_configs, Sequence), f"{type(step_configs)=}"
        self._step_configs = list(step_configs)
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self._steps: List[PipelineComponent] | None = None
        self._built = False

    @property
    def name(self) -> str:
        return self.PIPELINE_NAME

    def build(self, force: bool = False) -> None:
        if self._built:
            return
        if self._steps is None:
            self._steps = [build_from_config(config) for config in self._step_configs]
        for step in self._steps:
            step.build(force=force)
        self._built = True

    def check_inputs(self) -> None:
        assert (
            self._built
        ), f"Pipeline {self.PIPELINE_NAME} must be built before checking inputs"
        assert (
            self._steps is not None
        ), "Pipeline steps must be initialized during build"
        if len(self._steps) > 0:
            self._steps[0].check_inputs()

    def check_outputs(self) -> bool:
        assert (
            self._built
        ), f"Pipeline {self.PIPELINE_NAME} must be built before checking outputs"
        assert (
            self._steps is not None
        ), "Pipeline steps must be initialized during build"
        statuses = [component.check_outputs() for component in self._steps]
        return all(statuses)

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        assert self._built, f"Pipeline {self.PIPELINE_NAME} must be built before run"
        assert (
            self._steps is not None
        ), "Pipeline steps must be initialized during build"
        steps = self._steps
        total_steps = len(self._step_configs)
        logging.info(
            "Pipeline %s starting with %d steps (force=%s)",
            self.name,
            total_steps,
            force,
        )
        for index, component in enumerate(steps):
            component.build(force=force)
            logging.info(
                "[%d/%d] Launching step %s",
                index + 1,
                total_steps,
                component.name,
            )
            kwargs = component.run(kwargs, force=force)
            logging.info(
                "[%d/%d] Step %s completed",
                index + 1,
                total_steps,
                component.name,
            )
        logging.info("Pipeline %s finished", self.name)
        return kwargs

    def request(self, name: str) -> Dict[str, Any]:
        """Backward-style request that resolves graph dependencies by output file."""
        # Input validations
        assert isinstance(name, str), f"{type(name)=}"
        assert len(name) > 0, "name must be a non-empty string"

        self.build(force=False)
        target_output_path = self._resolve_requested_output_path(name=name)
        assert target_output_path is not None, (
            f"Requested file must be declared in step.output_files, got {name=}"
        )
        if target_output_path.exists():
            return {}

        step_state: Dict[str, Any] = {"requested_file": name}
        active_requests: List[str] = []
        step_state = self._ensure_requested_output(
            name=name,
            step_state=step_state,
            active_requests=active_requests,
        )
        assert isinstance(step_state, dict), f"{type(step_state)=}"
        if target_output_path.exists():
            return {}

        assert target_output_path.exists(), (
            f"Request target not produced: {target_output_path}"
        )
        return {}

    def _resolve_requested_output_path(self, name: str) -> Optional[Path]:
        # Input validations
        assert isinstance(name, str), f"{type(name)=}"
        assert len(name) > 0, "name must be a non-empty string"

        for component in self.steps:
            if self._component_contains_output(component=component, name=name):
                return self._resolve_component_output_path(
                    component=component,
                    name=name,
                )
        return None

    def _ensure_requested_output(
        self,
        name: str,
        step_state: Dict[str, Any],
        active_requests: List[str],
    ) -> Dict[str, Any]:
        # Input validations
        assert isinstance(name, str), f"{type(name)=}"
        assert len(name) > 0, "name must be a non-empty string"
        assert isinstance(step_state, dict), f"{type(step_state)=}"
        assert isinstance(active_requests, list), f"{type(active_requests)=}"
        assert all(
            isinstance(active_request, str) for active_request in active_requests
        ), f"{active_requests=}"

        target_output_path = self._resolve_requested_output_path(name=name)
        assert target_output_path is not None, (
            f"Requested file must be declared in step.output_files, got {name=}"
        )
        if target_output_path.exists():
            return step_state

        assert name not in active_requests, (
            f"Cycle detected while resolving requested file dependencies: "
            f"{active_requests + [name]}"
        )
        target_component_idx = self._find_component_index_for_output(name=name)
        assert target_component_idx is not None, (
            f"Requested file must be declared in step.output_files, got {name=}"
        )
        target_component = self.steps[target_component_idx]
        required_input_files = self._collect_request_input_files(name=name)

        active_requests.append(name)
        for required_input_file in required_input_files:
            producer_idx = self._find_component_index_for_output(name=required_input_file)
            if producer_idx is None:
                continue
            assert producer_idx != target_component_idx, (
                f"Step {target_component.name} cannot declare its own output as input; "
                f"{required_input_file=}"
            )
            step_state = self._ensure_requested_output(
                name=required_input_file,
                step_state=step_state,
                active_requests=active_requests,
            )

        if isinstance(target_component, BasePipeline):
            target_component.request(name=name)
        else:
            step_state = target_component.run(step_state, force=False)

        active_requests.pop()
        assert target_output_path.exists(), f"Failed to produce requested file: {name=}"
        return step_state

    def _component_contains_output(self, component: "PipelineComponent", name: str) -> bool:
        # Input validations
        assert isinstance(name, str), f"{type(name)=}"
        assert len(name) > 0, "name must be a non-empty string"

        if isinstance(component, BasePipeline):
            component.build(force=False)
            for child_component in component.steps:
                if self._component_contains_output(
                    component=child_component,
                    name=name,
                ):
                    return True
            return False

        for output_relpath in component.output_files:
            assert isinstance(output_relpath, str), f"{type(output_relpath)=}"
            if output_relpath == name:
                return True
        return False

    def _resolve_component_output_path(
        self,
        component: "PipelineComponent",
        name: str,
    ) -> Optional[Path]:
        # Input validations
        assert isinstance(name, str), f"{type(name)=}"
        assert len(name) > 0, "name must be a non-empty string"

        if isinstance(component, BasePipeline):
            component.build(force=False)
            for child_component in component.steps:
                resolved_path = self._resolve_component_output_path(
                    component=child_component,
                    name=name,
                )
                if resolved_path is not None:
                    return resolved_path
            return None

        for output_relpath in component.output_files:
            assert isinstance(output_relpath, str), f"{type(output_relpath)=}"
            if output_relpath == name:
                return component.output_root / output_relpath
        return None

    def _find_component_index_for_output(self, name: str) -> Optional[int]:
        # Input validations
        assert isinstance(name, str), f"{type(name)=}"
        assert len(name) > 0, "name must be a non-empty string"

        matching_indices = [
            step_idx
            for step_idx, component in enumerate(self.steps)
            if self._component_contains_output(component=component, name=name)
        ]
        assert len(matching_indices) <= 1, (
            f"Ambiguous requested file; produced by multiple steps: {name=}"
        )
        if len(matching_indices) == 0:
            return None
        return matching_indices[0]

    def _collect_request_input_files(self, name: str) -> List[str]:
        # Input validations
        assert isinstance(name, str), f"{type(name)=}"
        assert len(name) > 0, "name must be a non-empty string"

        target_component_idx = self._find_component_index_for_output(name=name)
        assert target_component_idx is not None, (
            f"Requested file must be declared in step.output_files, got {name=}"
        )
        target_component = self.steps[target_component_idx]
        if isinstance(target_component, BasePipeline):
            return target_component._collect_request_input_files(name=name)

        target_component.build(force=False)
        assert isinstance(target_component.input_files, list), (
            f"input_files must be initialized for step {target_component.name}"
        )
        assert all(
            isinstance(input_file, str) for input_file in target_component.input_files
        ), f"{target_component.input_files=}"
        return list(target_component.input_files)

    @property
    def steps(self) -> List["PipelineComponent"]:
        assert self._built, f"Pipeline {self.PIPELINE_NAME} must be built before access"
        assert (
            self._steps is not None
        ), "Pipeline steps must be initialized during build"
        return self._steps


PipelineComponent = BaseStep | BasePipeline
