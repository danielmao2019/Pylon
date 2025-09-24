from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import os
import glob
import torch

from agents.manager.progress_info import ProgressInfo
from agents.manager.base_job import BaseJob
from utils.io.config import load_config
from utils.io.json import load_json, save_json
from utils.builders.builder import build_from_config


class TrainingJob(BaseJob):
    """Copied trainer logic as classmethods for manager jobs."""

    EXPECTED_FILES = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
    LOG_PATTERN = "train_val*.log"

    # ====================================================================================================
    # 
    # ====================================================================================================

    def get_progress(self, force_progress_recompute: bool = False) -> ProgressInfo:
        basic_progress = self.get_session_progress(
            self.work_dir,
            self.get_expected_files(),
            force_progress_recompute=force_progress_recompute,
        )

        assert isinstance(self.config_dict, dict)
        assert 'epochs' in self.config_dict
        total_epochs = self.config_dict['epochs']

        return ProgressInfo(
            completed_epochs=basic_progress.completed_epochs,
            progress_percentage=basic_progress.progress_percentage,
            early_stopped=False,
            early_stopped_at_epoch=None,
            runner_type='trainer',
            total_epochs=total_epochs,
        )

    @classmethod
    def get_session_progress(
        cls,
        work_dir: str,
        expected_files: List[str],
        force_progress_recompute: bool = False,
    ) -> ProgressInfo:
        progress_file = os.path.join(work_dir, "progress.json")
        if not force_progress_recompute and os.path.exists(progress_file):
            data = load_json(progress_file)
            return ProgressInfo(**data)
        return cls._compute_and_cache_progress(work_dir, expected_files, force_progress_recompute)

    @classmethod
    def _compute_and_cache_progress(
        cls,
        work_dir: str,
        expected_files: List[str],
        force_progress_recompute: bool = False,
    ) -> ProgressInfo:
        progress_file = os.path.join(work_dir, "progress.json")
        if not force_progress_recompute and os.path.exists(progress_file):
            data = load_json(progress_file)
            return ProgressInfo(**data)

        completed_epochs = 0
        while True:
            if not cls._check_epoch_finished(
                epoch_dir=os.path.join(work_dir, f"epoch_{completed_epochs}"),
                expected_files=expected_files,
                check_load=False,
            ):
                break
            completed_epochs += 1

        config_path = BaseJob.get_config_filepath(work_dir)
        config = load_config(config_path)
        tot_epochs = config['epochs']
        early_stopping_config = config.get('early_stopping')

        early_stopped = False
        early_stopped_at_epoch: Optional[int] = None

        if early_stopping_config and completed_epochs < tot_epochs:
            early_stopped, early_stopped_at_epoch = cls._detect_early_stopping(
                work_dir, expected_files, config, completed_epochs
            )

        progress_data = ProgressInfo(
            completed_epochs=completed_epochs,
            progress_percentage=100.0 if early_stopped else (completed_epochs / tot_epochs * 100.0),
            early_stopped=early_stopped,
            early_stopped_at_epoch=early_stopped_at_epoch,
            runner_type='trainer',
            total_epochs=tot_epochs,
        )
        save_json(progress_data, progress_file)
        return progress_data

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

    def get_epoch_last_update(self) -> Optional[float]:
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
