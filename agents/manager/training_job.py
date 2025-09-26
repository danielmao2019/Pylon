from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import glob
import logging
import os
import torch

from agents.manager.progress_info import ProgressInfo
from agents.manager.default_job import DefaultJob
from agents.manager.job_types import RunnerKind
from utils.io.json import load_json, save_json
from utils.builders.builder import build_from_config


class TrainingJob(DefaultJob):
    """Copied trainer logic as classmethods for manager jobs."""

    EXPECTED_FILES = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
    LOG_PATTERN = "train_val*.log"
    runner_kind = RunnerKind.TRAINER

    # ====================================================================================================
    # 
    # ====================================================================================================

    def compute_progress(self) -> ProgressInfo:
        """Return cached progress if available, otherwise recompute and cache."""
        progress_file = os.path.join(self.work_dir, "progress.json")
        if not self.runtime.force_progress_recompute and os.path.exists(progress_file):
            try:
                data = load_json(progress_file)
            except RuntimeError as exc:
                logging.warning(
                    "Failed to load cached trainer progress from %s: %s. Recomputing progress.",
                    progress_file,
                    exc,
                )
            else:
                return ProgressInfo(**data)

        completed_epochs = 0

        while True:
            epoch_dir = os.path.join(self.work_dir, f"epoch_{completed_epochs}")
            if not self._check_epoch_finished(
                epoch_dir=epoch_dir,
                expected_files=self.EXPECTED_FILES,
                check_load=False,
            ):
                break
            completed_epochs += 1

        config = self.config_dict
        tot_epochs = config['epochs']
        early_stopping_config = config.get('early_stopping')

        early_stopped = False
        early_stopped_at_epoch: Optional[int] = None

        if early_stopping_config and completed_epochs < tot_epochs:
            early_stopped, early_stopped_at_epoch = self._detect_early_stopping(
                self.work_dir,
                self.EXPECTED_FILES,
                config,
                completed_epochs,
            )

        progress = ProgressInfo(
            completed_epochs=completed_epochs,
            progress_percentage=100.0 if early_stopped else (completed_epochs / tot_epochs * 100.0),
            early_stopped=early_stopped,
            early_stopped_at_epoch=early_stopped_at_epoch,
            runner_type=self.runner_kind.value,
            total_epochs=tot_epochs,
        )
        save_json(progress.to_dict(), progress_file)
        return progress

    # ------------------------------------------------------------------
    # Completion semantics
    # ------------------------------------------------------------------

    def is_complete(
        self,
        progress: ProgressInfo,
    ) -> bool:
        if progress.early_stopped:
            return True

        runtime = self.runtime
        target_epochs = runtime.epochs or progress.total_epochs or self.config_dict.get('epochs')
        try:
            target_int = int(target_epochs) if target_epochs is not None else 0
        except (TypeError, ValueError):
            target_int = 0

        if target_int <= 0:
            return False
        return progress.completed_epochs >= target_int

    @classmethod
    def _check_epoch_finished(
        cls,
        epoch_dir: str,
        expected_files: List[str],
        check_load: Optional[bool] = True,
    ) -> bool:
        if not os.path.isdir(epoch_dir):
            return False
        return all([
            os.path.isfile(os.path.join(epoch_dir, filename)) and
            os.path.getsize(os.path.join(epoch_dir, filename)) > 0 and
            ((not check_load) or cls._check_file_loadable(os.path.join(epoch_dir, filename)))
            for filename in expected_files
        ])

    @classmethod
    def _check_file_loadable(cls, filepath: str) -> bool:
        assert os.path.isfile(filepath)
        assert filepath.endswith(".json") or filepath.endswith(".pt")
        if filepath.endswith(".json"):
            try:
                load_json(filepath)
                return True
            except:
                return False
        elif filepath.endswith(".pt"):
            try:
                _ = torch.load(filepath)
                return True
            except:
                return False
        else:
            assert 0

    @classmethod
    def _detect_early_stopping(
        cls,
        work_dir: str,
        expected_files: List[str],
        config: Dict,
        completed_epochs: int,
    ) -> Tuple[bool, Optional[int]]:
        metric = build_from_config(config['metric']) if config.get('metric') else None
        if metric is None:
            return False, None

        early_stopping_config = config['early_stopping']
        early_stopping = build_from_config(
            config=early_stopping_config,
            work_dir=work_dir,
            tot_epochs=config['epochs'],
            metric=metric,
            expected_files=expected_files,
            logger=None,
        )

        if completed_epochs > early_stopping.patience:
            last_epoch = completed_epochs - 1
            if early_stopping.was_triggered_at_epoch(last_epoch):
                return True, last_epoch + 1
        return False, None

    # ====================================================================================================
    # 
    # ====================================================================================================

    def get_log_last_update(self) -> Optional[float]:
        """Get the timestamp of the last log update."""
        if not os.path.isdir(self.work_dir):
            return None
        logs = glob.glob(os.path.join(self.work_dir, self.LOG_PATTERN))
        if not logs:
            return None
        return max(os.path.getmtime(fp) for fp in logs)

    def get_artifact_last_update(self) -> Optional[float]:
        """Get the timestamp of the last epoch file update."""
        if not os.path.isdir(self.work_dir):
            return None
        epoch_dirs = glob.glob(os.path.join(self.work_dir, 'epoch_*'))
        if not epoch_dirs:
            return None
        timestamps = [
            os.path.getmtime(os.path.join(epoch_dir, filename))
            for epoch_dir in epoch_dirs
            for filename in self.EXPECTED_FILES
            if os.path.isfile(os.path.join(epoch_dir, filename))
        ]
        return max(timestamps) if timestamps else None
