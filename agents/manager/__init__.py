"""Agent manager package."""

from agents.manager.base_job import BaseJob
from agents.manager.base_progress_info import BaseProgressInfo
from agents.manager.default_job import DefaultJob, DefaultJobProgressInfo
from agents.manager.job_types import (
    RunnerKind,
    register_runner_kind,
    registered_runner_kinds,
)
from agents.manager.manager import (
    Manager,
    register_job_class,
    register_job_classes,
    register_runner_detector,
    register_runner_detectors,
    registered_job_classes,
)
from agents.manager.nerfstudio_job import NerfStudioJob
from agents.manager.runtime import JobRuntimeParams
from agents.monitor.process_info import ProcessInfo

__all__ = [
    "BaseJob",
    "BaseProgressInfo",
    "DefaultJob",
    "DefaultJobProgressInfo",
    "RunnerKind",
    "register_runner_kind",
    "registered_runner_kinds",
    "Manager",
    "register_job_class",
    "register_job_classes",
    "register_runner_detector",
    "register_runner_detectors",
    "registered_job_classes",
    "NerfStudioJob",
    "JobRuntimeParams",
    "ProcessInfo",
]
