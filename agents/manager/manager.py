from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence, Type

from agents.manager.base_job import BaseJob
from agents.manager.default_job import DefaultJob
from agents.manager.training_job import TrainingJob
from agents.manager.evaluation_job import EvaluationJob
from agents.manager.nerfstudio_job import NerfStudioJob
from agents.manager.job_types import RunnerKind
from agents.manager.runtime import JobRuntimeParams
from utils.io.config import load_config
from agents.monitor.gpu_status import GPUStatus
from agents.monitor.process_info import ProcessInfo
from agents.monitor.system_monitor import SystemMonitor


class Manager:
    """Builds BaseJob instances for a collection of configs."""

    DEFAULT_JOB_CLASSES: Dict[RunnerKind, Type[DefaultJob]] = {
        RunnerKind.TRAINER: TrainingJob,
        RunnerKind.EVALUATOR: EvaluationJob,
        RunnerKind.NERFSTUDIO: NerfStudioJob,
    }

    def __init__(
        self,
        commands: List[str],
        epochs: int,
        system_monitors: Dict[str, SystemMonitor],
        sleep_time: int = 86400,
        outdated_days: int = 30,
        force_progress_recompute: bool = False,
    ) -> None:
        assert isinstance(commands, list)
        assert isinstance(epochs, int)
        assert isinstance(system_monitors, dict)
        assert all(isinstance(monitor, SystemMonitor) for monitor in system_monitors.values())

        self.commands = commands
        self.epochs = epochs
        self.system_monitors = system_monitors
        self.sleep_time = sleep_time
        self.outdated_days = outdated_days
        self.force_progress_recompute = force_progress_recompute
        self.job_classes = dict(self.DEFAULT_JOB_CLASSES)

    def build_jobs(self) -> Dict[str, DefaultJob]:
        """Build BaseJob instances for all configs."""
        all_connected_gpus = [
            gpu
            for monitor in self.system_monitors.values()
            for gpu in monitor.connected_gpus
        ]
        command_to_process_info = self.build_command_to_process_mapping(all_connected_gpus)

        runtime = JobRuntimeParams(
            epochs=self.epochs,
            sleep_time=self.sleep_time,
            outdated_days=self.outdated_days,
            command_processes=command_to_process_info,
            force_progress_recompute=self.force_progress_recompute,
        )

        def _construct(command: str) -> DefaultJob:
            runner_kind = self._detect_runner_type(command)
            job_cls = self.job_classes.get(runner_kind)
            if job_cls is None:
                raise KeyError(f"No job class registered for runner {runner_kind!r}")
            job = job_cls(command)
            job.configure(runtime)
            return job

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_construct, self.commands))

        jobs = dict(zip(self.commands, results))
        return jobs

    def _detect_runner_type(self, command: str) -> RunnerKind:
        command_result = self._detect_from_command(command)
        if command_result is not None:
            return command_result

        config_filepath = self._extract_config_filepath(command)

        config_result = self._detect_from_config(config_filepath)
        if config_result is not None:
            return config_result

        work_dir = self._workdir_from_config(config_filepath)

        artifact_result = self._detect_from_artifacts(work_dir)
        if artifact_result is not None:
            return artifact_result

        raise ValueError(
            f"Unable to determine runner type for command: {command!r}."
        )

    @staticmethod
    def _detect_from_command(command: str) -> RunnerKind | None:
        if 'ns-train' in command:
            return RunnerKind.NERFSTUDIO
        return None

    @staticmethod
    def _extract_config_filepath(command: str) -> str:
        tokens = [token for token in command.split() if token]
        for token in tokens:
            if token.startswith('--config-filepath='):
                return token.split('=', 1)[1]
        idx = tokens.index('--config-filepath')
        return tokens[idx + 1]

    @staticmethod
    def _workdir_from_config(config_filepath: str) -> str:
        return config_filepath.replace('configs', 'logs')

    @staticmethod
    def _detect_from_artifacts(config_workdir: str) -> RunnerKind | None:
        eval_scores = os.path.join(config_workdir, 'evaluation_scores.json')
        if os.path.isfile(eval_scores):
            return RunnerKind.EVALUATOR

        epoch_dir = os.path.join(config_workdir, 'epoch_0')
        if os.path.isdir(epoch_dir) and os.path.isfile(os.path.join(epoch_dir, 'validation_scores.json')):
            return RunnerKind.TRAINER

        return None

    @staticmethod
    def _detect_from_config(config_filepath: str) -> RunnerKind | None:
        if not os.path.exists(config_filepath):
            return None
        try:
            config = load_config(config_filepath)
        except Exception:
            return None

        runner = config.get('runner')
        if isinstance(runner, type):
            name = runner.__name__.lower()
            if 'evaluat' in name:
                return RunnerKind.EVALUATOR
            if 'train' in name or 'trainer' in name:
                return RunnerKind.TRAINER

        return None

    @staticmethod
    def build_command_to_process_mapping(
        connected_gpus: List[GPUStatus],
    ) -> Dict[str, ProcessInfo]:
        """Build mapping from runtime command string to ProcessInfo."""
        command_to_process: Dict[str, ProcessInfo] = {}
        for gpu in connected_gpus:
            for process in gpu.processes:
                assert isinstance(process, ProcessInfo), (
                    f'Expected ProcessInfo instance, got {type(process)}'
                )
                command_to_process[process.cmd] = process
        return command_to_process
