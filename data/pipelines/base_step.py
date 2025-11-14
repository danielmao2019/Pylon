"""Base step abstraction for simple pipelines."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
import os
import subprocess
import sys
from typing import ClassVar


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class BaseStep(ABC):
    """Abstract step that defines the common pipeline contract."""

    STEP_NAME: ClassVar[str] = "base_step"
    INPUT_FILES: ClassVar[list[str]] = []
    OUTPUT_FILES: ClassVar[list[str]] = []

    def __init__(self, input_root: str | Path, output_root: str | Path):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)

    @property
    def name(self) -> str:
        return self.STEP_NAME

    def check_inputs(self) -> None:
        missing = [
            self.input_root / rel
            for rel in self.INPUT_FILES
            if not (self.input_root / rel).exists()
        ]
        if missing:
            missing_str = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Cannot run step {self.STEP_NAME}: missing inputs: {missing_str}"
            )

    def check_outputs(self) -> bool:
        """Report output status using declared ``OUTPUT_FILES`` when available."""
        assert isinstance(self.OUTPUT_FILES, list), (
            f"OUTPUT_FILES must be list for step {self.STEP_NAME}, "
            f"got {type(self.OUTPUT_FILES)}"
        )
        assert all(isinstance(entry, str) for entry in self.OUTPUT_FILES), (
            f"OUTPUT_FILES items must be str for step {self.STEP_NAME}, "
            f"got types={{{', '.join(sorted({type(entry).__name__ for entry in self.OUTPUT_FILES}))}}}"
        )
        missing = [
            self.output_root / rel
            for rel in self.OUTPUT_FILES
            if not (self.output_root / rel).exists()
        ]
        return not missing

    @abstractmethod
    def run(self, force: bool = False) -> None:
        """Execute the step, optionally forcing a re-run."""

    def _run_shell(self, command: str) -> None:
        """Utility to run shell commands with logging."""
        supports_color = (
            hasattr(sys.stderr, "isatty")
            and sys.stderr.isatty()
            and os.environ.get("TERM") not in {"", "dumb"}
        )
        prefix, suffix = ("\033[36m", "\033[0m") if supports_color else ("", "")
        logging.info("%sRunning command: %s%s", prefix, command, suffix)
        subprocess.run(["bash", "-lc", command], check=True)
