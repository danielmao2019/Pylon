"""Base step abstraction for simple pipelines."""

import logging
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class BaseStep(ABC):
    """Abstract step that defines the common pipeline contract."""

    STEP_NAME: ClassVar[str] = "base_step"

    def __init__(self, input_root: str | Path, output_root: str | Path):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.input_files: List[str] | None = None
        self.output_files: List[str] | None = None
        self._built = False

    @property
    def name(self) -> str:
        return self.STEP_NAME

    def build(self, force: bool = False) -> None:
        self._init_input_files()
        assert isinstance(self.input_files, list), (
            f"input_files must be list for step {self.STEP_NAME}, "
            f"got {type(self.input_files)}"
        )
        assert all(isinstance(entry, str) for entry in self.input_files), (
            f"input_files entries must be str for step {self.STEP_NAME}, "
            f"got types={{{', '.join(sorted({type(entry).__name__ for entry in self.input_files}))}}}"
        )
        self._init_output_files()
        assert isinstance(self.output_files, list), (
            f"output_files must be list for step {self.STEP_NAME}, "
            f"got {type(self.output_files)}"
        )
        assert all(isinstance(entry, str) for entry in self.output_files), (
            f"output_files entries must be str for step {self.STEP_NAME}, "
            f"got types={{{', '.join(sorted({type(entry).__name__ for entry in self.output_files}))}}}"
        )
        self._built = True

    def check_inputs(self) -> None:
        assert (
            self._built
        ), f"Step {self.STEP_NAME} must be built before checking inputs"
        assert isinstance(
            self.input_files, list
        ), "input_files must be initialized during build"
        missing = [
            self.input_root / rel
            for rel in self.input_files
            if not (self.input_root / rel).exists()
        ]
        if missing:
            missing_str = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Cannot run step {self.STEP_NAME}: missing inputs: {missing_str}"
            )

    def check_outputs(self) -> bool:
        """Report output status using declared ``output_files`` when available."""
        assert (
            self._built
        ), f"Step {self.STEP_NAME} must be built before checking outputs"
        assert isinstance(
            self.output_files, list
        ), "output_files must be initialized during build"
        missing = [
            self.output_root / rel
            for rel in self.output_files
            if not (self.output_root / rel).exists()
        ]
        return not missing

    def _init_input_files(self) -> None:
        self.input_files = []

    def _init_output_files(self) -> None:
        self.output_files = []

    @abstractmethod
    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
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
