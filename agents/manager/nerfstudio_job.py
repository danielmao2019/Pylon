from __future__ import annotations

import os
import shlex

from agents.manager.base_job import BaseJob
from agents.manager.progress_info import ProgressInfo


class NerfStudioJob(BaseJob):
    """Minimal job wrapper for NeRFStudio commands."""

    EXPECTED_FILES = [
        "config.yml",
        "dataparser_transforms.json",
        os.path.join("checkpoints", "step-000029999.ckpt"),
    ]

    def __init__(self, command: str) -> None:
        super().__init__(command=command)

    def derive_work_dir(self) -> str:
        try:
            tokens = shlex.split(self.command)
        except ValueError:
            tokens = self.command.split()

        lookup_flags = (
            '--output-dir', '--output_dir', '--output',
            '--workspace', '--run-dir', '--run_dir', '--logdir', '--log-dir'
        )
        for flag in lookup_flags:
            if flag in tokens:
                idx = tokens.index(flag)
                if idx + 1 < len(tokens):
                    candidate = tokens[idx + 1]
                    if candidate:
                        return os.path.normpath(os.path.expanduser(candidate))
            prefix = f"{flag}="
            for token in tokens:
                if token.startswith(prefix) and len(token) > len(prefix):
                    return os.path.normpath(os.path.expanduser(token[len(prefix):]))

        raise ValueError(
            "Unable to determine work directory for NerfStudioJob. "
            "Provide a supported command flag (e.g. --output-dir) in the command."
        )

    def compute_progress(self) -> ProgressInfo:
        has_expected_files = all(
            os.path.isfile(os.path.join(self.work_dir, relative_path))
            for relative_path in self.EXPECTED_FILES
        )

        return ProgressInfo(
            completed_epochs=1 if has_expected_files else 0,
            progress_percentage=100.0 if has_expected_files else 0.0,
            early_stopped=False,
            early_stopped_at_epoch=None,
            runner_type='nerfstudio',
            total_epochs=1,
        )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        return self.process_info is not None

    def is_complete(self, progress: ProgressInfo) -> bool:
        return progress.completed_epochs >= 1

    def is_outdated(self) -> bool:
        return False

    def is_stuck(self) -> bool:
        return False

    def is_failed(self, progress: ProgressInfo) -> bool:
        return not self.is_complete(progress)
