from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List

from agents.manager.default_job import DefaultJob
from agents.manager.training_job import TrainingJob
from agents.manager.evaluation_job import EvaluationJob
from utils.io.config import load_config
from agents.monitor.gpu_status import GPUStatus
from agents.monitor.process_info import ProcessInfo
from agents.monitor.system_monitor import SystemMonitor


class Manager:
    """Builds BaseJob instances for a collection of configs."""

    def __init__(
        self,
        config_files: List[str],
        epochs: int,
        system_monitors: Dict[str, SystemMonitor],
        sleep_time: int = 86400,
        outdated_days: int = 30,
        force_progress_recompute: bool = False,
    ) -> None:
        assert isinstance(config_files, list)
        assert isinstance(epochs, int)
        assert isinstance(system_monitors, dict)
        assert all(isinstance(monitor, SystemMonitor) for monitor in system_monitors.values())

        self.config_files = config_files
        self.epochs = epochs
        self.system_monitors = system_monitors
        self.sleep_time = sleep_time
        self.outdated_days = outdated_days
        self.force_progress_recompute = force_progress_recompute

    def _detect_runner_type(self, work_dir: str, config: dict | None) -> str:
        """Detect runner type from work_dir structure or config.

        Moved from tracker to manager without modification in logic.
        """
        import os
        # Strategy 1: Check existing files (based on eval_viewer's proven approach)
        # Check for BaseEvaluator pattern: evaluation_scores.json directly in work_dir
        if os.path.exists(os.path.join(work_dir, "evaluation_scores.json")):
            return 'evaluator'

        # Check for BaseTrainer pattern: epoch folders with validation_scores.json
        epoch_0_dir = os.path.join(work_dir, "epoch_0")
        validation_scores_path = os.path.join(epoch_0_dir, "validation_scores.json")
        if os.path.exists(epoch_0_dir) and os.path.exists(validation_scores_path):
            return 'trainer'

        # Strategy 2: Check config if available (additional context)
        if config:
            assert 'runner' in config, f"Config must have 'runner' key, got keys: {list(config.keys())}"
            runner_config = config['runner']
            # Enforce contract: runner config must be a direct class reference
            assert isinstance(runner_config, type), (
                f"Expected runner to be a class, got {type(runner_config)}: {runner_config}"
            )
            
            # Infer by class name convention (kept identical to original behavior if any)
            name = runner_config.__name__.lower()
            if 'evaluat' in name:
                return 'evaluator'
            if 'train' in name:
                return 'trainer'

        # Fail fast if cannot determine
        raise ValueError(
            f"Unable to determine runner type for work_dir={work_dir!r}. "
            f"Expected evaluator or trainer artifacts."
        )

    def build_jobs(self) -> Dict[str, DefaultJob]:
        """Build BaseJob instances for all configs."""
        all_connected_gpus = [
            gpu
            for monitor in self.system_monitors.values()
            for gpu in monitor.connected_gpus
        ]
        config_to_process_info = self.build_config_to_process_mapping(all_connected_gpus)

        def _construct(config: str) -> DefaultJob:
            work_dir = DefaultJob.get_work_dir(config)
            config_dict = load_config(config)
            runner = self._detect_runner_type(work_dir, config_dict)
            if runner == 'evaluator':
                job: DefaultJob = EvaluationJob(config)  # type: ignore[assignment]
            else:
                job = TrainingJob(config)  # type: ignore[assignment]
            job.populate(
                epochs=self.epochs,
                config_to_process_info=config_to_process_info,
                sleep_time=self.sleep_time,
                outdated_days=self.outdated_days,
                force_progress_recompute=self.force_progress_recompute,
            )
            return job

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_construct, self.config_files))

        jobs = dict(zip(self.config_files, results))
        assert set(jobs.keys()) == {job.config_filepath for job in jobs.values()}, (
            "Mismatch between config files and built job configs"
        )
        return jobs

    @staticmethod
    def build_config_to_process_mapping(
        connected_gpus: List[GPUStatus],
    ) -> Dict[str, ProcessInfo]:
        """Build mapping from config file to ProcessInfo for running experiments."""
        config_to_process: Dict[str, ProcessInfo] = {}
        for gpu in connected_gpus:
            for process in gpu.processes:
                assert isinstance(process, ProcessInfo), (
                    f'Expected ProcessInfo instance, got {type(process)}'
                )
                if 'python main.py --config-filepath' not in process.cmd:
                    continue
                config = DefaultJob.parse_config(process.cmd)
                config_to_process[config] = process
        return config_to_process


__all__ = ['Manager']
