"""Simple sequential pipeline runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Sequence

from project.pipelines.base_step import BaseStep


class BasePipeline(BaseStep):
    """Executes a list of steps sequentially and supports nesting."""

    STEP_NAME: ClassVar[str] = "base_pipeline"

    def __init__(
        self,
        steps: Sequence[BaseStep],
        input_root: str | Path,
        output_root: str | Path,
    ) -> None:
        assert steps, "Pipeline must receive at least one step"
        super().__init__(
            input_root=input_root,
            output_root=output_root,
        )
        self.steps = list(steps)

    def run(self, force: bool = False) -> None:
        total_steps = len(self.steps)
        logging.info(
            "Pipeline %s starting with %d steps (force=%s)",
            self.name,
            total_steps,
            force,
        )
        for index, step in enumerate(self.steps):
            logging.info("[%d/%d] Launching step %s", index + 1, total_steps, step.name)
            step.run(force=force)
            logging.info("[%d/%d] Step %s completed", index + 1, total_steps, step.name)
        logging.info("Pipeline %s finished", self.name)

    def check_outputs(self) -> bool:
        assert self.steps, "Pipeline must define steps before checking outputs"
        statuses = [step.check_outputs() for step in self.steps]
        completion = int(all(statuses))
        return bool(completion)
