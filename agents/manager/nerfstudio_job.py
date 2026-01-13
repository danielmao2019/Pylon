import os
import shlex
from dataclasses import dataclass

from agents.manager.base_progress_info import BaseProgressInfo
from agents.manager.base_job import BaseJob
from agents.manager.runtime import JobRuntimeParams
from agents.manager.job_types import RunnerKind


@dataclass
class NerfStudioProgressInfo(BaseProgressInfo):
    completed_iterations: int
    target_iterations: int


class NerfStudioJob(BaseJob):
    """Minimal job wrapper for NeRFStudio commands."""

    EXPECTED_FILES = [
        "config.yml",
        "dataparser_transforms.json",
        os.path.join("nerfstudio_models", "step-000029999.ckpt"),
    ]
    runner_kind = RunnerKind.NERFSTUDIO
    TARGET_ITERATIONS: int = 30000

    def derive_work_dir(self) -> str:
        tokens = shlex.split(self.command)

        lookup_flags = (
            '--output-dir',
            '--output_dir',
            '--output',
            '--workspace',
            '--run-dir',
            '--run_dir',
            '--logdir',
            '--log-dir',
        )
        for flag in lookup_flags:
            if flag in tokens:
                idx = tokens.index(flag)
                assert idx + 1 < len(
                    tokens
                ), f"Flag {flag} is missing a value in command {self.command!r}"
                candidate = tokens[idx + 1]
                assert candidate, f"Empty value provided for {flag}"
                return os.path.normpath(os.path.expanduser(candidate))
            prefix = f"{flag}="
            for token in tokens:
                if token.startswith(prefix) and len(token) > len(prefix):
                    candidate = token[len(prefix) :]
                    assert candidate, f"Empty value provided for {flag}"
                    return os.path.normpath(os.path.expanduser(candidate))

        raise ValueError(
            "Unable to determine work directory for NerfStudioJob. "
            "Provide a supported command flag (e.g. --output-dir) in the command."
        )

    def compute_progress(self, runtime: JobRuntimeParams) -> NerfStudioProgressInfo:
        model_dir = os.path.join(self.work_dir, "nerfstudio_models")
        checkpoint_steps = self._collect_checkpoint_steps(model_dir)
        completed_iterations = max(checkpoint_steps) if checkpoint_steps else 0
        percentage = (completed_iterations / self.TARGET_ITERATIONS) * 100.0

        return NerfStudioProgressInfo(
            completed_iterations=completed_iterations,
            target_iterations=self.TARGET_ITERATIONS,
            progress_percentage=min(percentage, 100.0),
        )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_active(self, runtime: JobRuntimeParams) -> bool:
        return self.process_info is not None

    def is_stuck(self, runtime: JobRuntimeParams) -> bool:
        return False

    def tmux_session_name(self) -> str:
        data_dir = self._extract_data_dir()
        basename = os.path.basename(os.path.normpath(data_dir))
        assert basename, f"Unable to derive session name from data dir {data_dir!r}"
        return f"ns-train-{basename}"

    def _extract_data_dir(self) -> str:
        tokens = shlex.split(self.command)

        flag = '--data'
        if flag in tokens:
            idx = tokens.index(flag)
            assert idx + 1 < len(
                tokens
            ), f"Flag {flag} is missing a value in command {self.command!r}"
            candidate = tokens[idx + 1]
            assert candidate, f"Empty value provided for {flag}"
            return os.path.expanduser(candidate)

        prefix = f"{flag}="
        for token in tokens:
            if token.startswith(prefix) and len(token) > len(prefix):
                candidate = token[len(prefix) :]
                assert candidate, f"Empty value provided for {flag}"
                return os.path.expanduser(candidate)

        raise ValueError("NerfStudio command must specify data directory via --data")

    @staticmethod
    def _collect_checkpoint_steps(model_dir: str) -> list[int]:
        if not os.path.isdir(model_dir):
            return []
        steps: list[int] = []
        for filename in os.listdir(model_dir):
            if not filename.startswith("step-") or not filename.endswith(".ckpt"):
                continue
            stem = filename[len("step-") : -len(".ckpt")]
            try:
                steps.append(int(stem))
            except ValueError:
                continue
        return steps
