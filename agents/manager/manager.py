from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Type

from agents.manager.base_job import BaseJob
from agents.manager.training_job import TrainingJob
from agents.manager.evaluation_job import EvaluationJob
from agents.manager.nerfstudio_job import NerfStudioJob
from project.agents.manager.nerfstudio_generation_job import (
    MultiServerNerfStudioGenerationJob,
)
from project.agents.manager.nerfstudio_data_job import NerfStudioDataJob
from project.agents.manager.las_to_ply_job import LasToPlyOffsetJob
from project.agents.manager.point_cloud_jobs import (
    DensePointCloudJob,
    SparsePointCloudJob,
)
from agents.manager.job_types import RunnerKind
from agents.manager.runtime import JobRuntimeParams
from utils.io.config import load_config
from agents.monitor.cpu_status import CPUStatus
from agents.monitor.gpu_status import GPUStatus
from agents.monitor.process_info import ProcessInfo
from agents.monitor.system_monitor import SystemMonitor
from runners.evaluators.base_evaluator import BaseEvaluator
from runners.trainers.base_trainer import BaseTrainer


class Manager:
    """Builds BaseJob instances for a collection of configs."""

    DEFAULT_JOB_CLASSES: Dict[RunnerKind, Type[BaseJob]] = {
        RunnerKind.TRAINER: TrainingJob,
        RunnerKind.EVALUATOR: EvaluationJob,
        RunnerKind.NERFSTUDIO: NerfStudioJob,
        RunnerKind.NERFSTUDIO_GENERATION: MultiServerNerfStudioGenerationJob,
        RunnerKind.NERFSTUDIO_DATA: NerfStudioDataJob,
        RunnerKind.LAS_TO_PLY: LasToPlyOffsetJob,
        RunnerKind.DENSE_POINT_CLOUD: DensePointCloudJob,
        RunnerKind.SPARSE_POINT_CLOUD: SparsePointCloudJob,
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
        assert all(
            isinstance(monitor, SystemMonitor) for monitor in system_monitors.values()
        )

        self.commands = commands
        self.epochs = epochs
        self.system_monitors = system_monitors
        self.sleep_time = sleep_time
        self.outdated_days = outdated_days
        self.force_progress_recompute = force_progress_recompute
        self.job_classes = dict(self.DEFAULT_JOB_CLASSES)

    def build_jobs(self) -> Dict[str, BaseJob]:
        """Build BaseJob instances for all configs."""
        all_connected_gpus = [
            gpu
            for monitor in self.system_monitors.values()
            for gpu in monitor.connected_gpus
        ]
        all_connected_cpus = [
            monitor.connected_cpu
            for monitor in self.system_monitors.values()
            if monitor.connected_cpu is not None
        ]
        command_to_process_info = self.build_command_to_process_mapping(
            all_connected_gpus,
            all_connected_cpus,
        )

        runtime = JobRuntimeParams(
            epochs=self.epochs,
            sleep_time=self.sleep_time,
            outdated_days=self.outdated_days,
            command_processes=command_to_process_info,
            force_progress_recompute=self.force_progress_recompute,
        )

        def _construct(command: str) -> BaseJob:
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
        self._jobs = list(jobs.values())
        return jobs

    def compute_average_progress(self) -> float:
        assert hasattr(self, '_jobs'), "Jobs have not been built yet"
        jobs = getattr(self, '_jobs')
        assert (
            jobs is not None and len(jobs) > 0
        ), "No jobs available to compute progress"

        assert all(
            job.progress is not None for job in jobs
        ), "Job progress must be computed for all jobs"

        total = sum(job.progress.progress_percentage for job in jobs)
        return total / len(jobs)

    def _detect_runner_type(self, command: str) -> RunnerKind:
        command_result = self._detect_from_command(command)
        if command_result is not None:
            return command_result

        config_filepath = self._extract_config_filepath_from_command(command)

        config_result = self._detect_from_config(config_filepath)
        if config_result is not None:
            return config_result

        work_dir = self._extract_work_dir_from_config_filepath(config_filepath)

        artifact_result = self._detect_from_artifacts(work_dir)
        if artifact_result is not None:
            return artifact_result

        raise ValueError(f"Unable to determine runner type for command: {command!r}.")

    @staticmethod
    def _detect_from_command(command: str) -> RunnerKind | None:
        if 'ns-train' in command:
            return RunnerKind.NERFSTUDIO
        if command.strip().startswith('python gen_ivision_mt_nerfstudio.py'):
            return RunnerKind.NERFSTUDIO_GENERATION
        if command.strip().startswith(
            'python project/scripts/3_prepare_nerfstudio_data/gen_nerfstudio_data.py'
        ):
            return RunnerKind.NERFSTUDIO_DATA
        if command.strip().startswith(
            'python project/scripts/1_coord_transforms/compute_las_to_ply_offsets.py'
        ):
            return RunnerKind.LAS_TO_PLY
        if command.strip().startswith(
            'python project/scripts/2_preprocess_point_clouds/process_dense_point_clouds.py'
        ):
            return RunnerKind.DENSE_POINT_CLOUD
        if command.strip().startswith(
            'python project/scripts/2_preprocess_point_clouds/process_sparse_point_clouds.py'
        ):
            return RunnerKind.SPARSE_POINT_CLOUD
        return None

    @staticmethod
    def _extract_config_filepath_from_command(command: str) -> str:
        assert "python main.py --config-filepath" in command
        tokens = [token for token in command.split() if token]
        idx = tokens.index('--config-filepath')
        return tokens[idx + 1]

    @staticmethod
    def _detect_from_config(config_filepath: str) -> RunnerKind | None:
        if not os.path.exists(config_filepath):
            return None
        try:
            config = load_config(config_filepath)
        except Exception:
            return None

        runner = config.get('runner')
        assert isinstance(runner, type)
        if issubclass(runner, BaseEvaluator):
            return RunnerKind.EVALUATOR
        if issubclass(runner, BaseTrainer):
            return RunnerKind.TRAINER

        return None

    @staticmethod
    def _extract_work_dir_from_config_filepath(config_filepath: str) -> str:
        normalized = os.path.normpath(config_filepath)
        rel_path = os.path.relpath(normalized, start='./configs')
        log_rel_path, _ = os.path.splitext(rel_path)
        return os.path.normpath(os.path.join('./logs', log_rel_path))

    @staticmethod
    def _detect_from_artifacts(config_workdir: str) -> RunnerKind | None:
        eval_scores = os.path.join(config_workdir, 'evaluation_scores.json')
        if os.path.isfile(eval_scores):
            return RunnerKind.EVALUATOR

        epoch_dir = os.path.join(config_workdir, 'epoch_0')
        # Use TrainingJob.EXPECTED_FILES for trainer artifact detection
        has_expected_artifact = any(
            os.path.isfile(os.path.join(epoch_dir, fname))
            for fname in TrainingJob.EXPECTED_FILES
        )
        if os.path.isdir(epoch_dir) and has_expected_artifact:
            return RunnerKind.TRAINER

        return None

    @staticmethod
    def build_command_to_process_mapping(
        connected_gpus: List[GPUStatus],
        connected_cpus: Optional[List[CPUStatus]] = None,
    ) -> Dict[str, ProcessInfo]:
        """Build mapping from runtime command string to ProcessInfo."""

        # Step 1: gather every process the monitors can see.
        all_processes: List[tuple[str, ProcessInfo]] = []
        for gpu in connected_gpus:
            for process in gpu.processes:
                assert isinstance(process, ProcessInfo)
                all_processes.append((gpu.server, process))

        for cpu in connected_cpus or []:
            for process in cpu.processes:
                assert isinstance(process, ProcessInfo)
                all_processes.append((cpu.server, process))

        # Step 2: deduplicate (server, pid) pairs – CPU/GPU monitors can report the same process.
        deduped_processes: List[tuple[str, ProcessInfo]] = []
        seen_process_ids: set[tuple[str, str]] = set()
        for server, process in all_processes:
            key = (server, process.pid)
            if key in seen_process_ids:
                continue
            seen_process_ids.add(key)
            deduped_processes.append((server, process))

        # Step 3: build the command-to-process mapping (first occurrence wins).
        command_to_process: Dict[str, ProcessInfo] = {}
        for _, process in deduped_processes:
            if process.cmd not in command_to_process:
                command_to_process[process.cmd] = process

        return command_to_process
