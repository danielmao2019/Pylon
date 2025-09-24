from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List

from agents.manager.base_job import BaseJob
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

    def build_jobs(self) -> Dict[str, BaseJob]:
        """Build BaseJob instances for all configs."""
        all_connected_gpus = [
            gpu
            for monitor in self.system_monitors.values()
            for gpu in monitor.connected_gpus
        ]
        config_to_process_info = self.build_config_to_process_mapping(all_connected_gpus)

        builder = partial(
            BaseJob.build,
            epochs=self.epochs,
            config_to_process_info=config_to_process_info,
            sleep_time=self.sleep_time,
            outdated_days=self.outdated_days,
            force_progress_recompute=self.force_progress_recompute,
        )

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(builder, self.config_files))

        jobs = dict(zip(self.config_files, results))
        assert set(jobs.keys()) == {job.config for job in jobs.values()}, (
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
                config = BaseJob.parse_config(process.cmd)
                config_to_process[config] = process
        return config_to_process


__all__ = ['Manager']
