"""Simple sequential pipeline runner."""

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence

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
        assert step_configs, "Pipeline must receive at least one component"
        self._step_configs = list(step_configs)
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self._steps: List[PipelineComponent] | None = None
        self._built = False

    @property
    def name(self) -> str:
        return self.PIPELINE_NAME

    def build(self, force: bool = False) -> None:
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

    @property
    def steps(self) -> List["PipelineComponent"]:
        assert self._built, f"Pipeline {self.PIPELINE_NAME} must be built before access"
        assert (
            self._steps is not None
        ), "Pipeline steps must be initialized during build"
        return self._steps


PipelineComponent = BaseStep | BasePipeline
