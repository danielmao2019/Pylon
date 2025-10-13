from __future__ import annotations

import os
import shlex

from agents.manager.base_job import BaseJob
from agents.manager.progress_info import ProgressInfo
from agents.manager.runtime import JobRuntimeParams
from agents.manager.job_types import RunnerKind


class NerfStudioJob(BaseJob):
    """Minimal job wrapper for NeRFStudio commands."""

    EXPECTED_FILES = [
        "config.yml",
        "dataparser_transforms.json",
        os.path.join("nerfstudio_models", "step-000029999.ckpt"),
    ]
    runner_kind = RunnerKind.NERFSTUDIO

    def derive_work_dir(self) -> str:
        tokens = shlex.split(self.command)

        lookup_flags = (
            '--output-dir', '--output_dir', '--output',
            '--workspace', '--run-dir', '--run_dir', '--logdir', '--log-dir'
        )
        for flag in lookup_flags:
            if flag in tokens:
                idx = tokens.index(flag)
                assert idx + 1 < len(tokens), (
                    f"Flag {flag} is missing a value in command {self.command!r}"
                )
                candidate = tokens[idx + 1]
                assert candidate, f"Empty value provided for {flag}"
                return os.path.normpath(os.path.expanduser(candidate))
            prefix = f"{flag}="
            for token in tokens:
                if token.startswith(prefix) and len(token) > len(prefix):
                    candidate = token[len(prefix):]
                    assert candidate, f"Empty value provided for {flag}"
                    return os.path.normpath(os.path.expanduser(candidate))

        raise ValueError(
            "Unable to determine work directory for NerfStudioJob. "
            "Provide a supported command flag (e.g. --output-dir) in the command."
        )

    def compute_progress(self, runtime: JobRuntimeParams) -> ProgressInfo:
        completed_run = False

        if os.path.isdir(self.work_dir):
            for root, _dirs, _files in os.walk(self.work_dir):
                if all(
                    os.path.isfile(os.path.join(root, relative_path))
                    for relative_path in self.EXPECTED_FILES
                ):
                    completed_run = True
                    break

        return ProgressInfo(
            completed_epochs=1 if completed_run else 0,
            progress_percentage=100.0 if completed_run else 0.0,
            early_stopped=False,
            early_stopped_at_epoch=None,
            total_epochs=1,
        )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_active(self, runtime: JobRuntimeParams) -> bool:
        return self.process_info is not None

    def is_complete(self, progress: ProgressInfo, runtime: JobRuntimeParams) -> bool:
        return progress.completed_epochs >= 1

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
            assert idx + 1 < len(tokens), (
                f"Flag {flag} is missing a value in command {self.command!r}"
            )
            candidate = tokens[idx + 1]
            assert candidate, f"Empty value provided for {flag}"
            return os.path.expanduser(candidate)

        prefix = f"{flag}="
        for token in tokens:
            if token.startswith(prefix) and len(token) > len(prefix):
                candidate = token[len(prefix):]
                assert candidate, f"Empty value provided for {flag}"
                return os.path.expanduser(candidate)

        raise ValueError(
            "NerfStudio command must specify data directory via --data"
        )
