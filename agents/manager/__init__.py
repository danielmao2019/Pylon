"""Agent manager package."""

from agents.manager.base_job import BaseJob
from agents.manager.base_progress_info import BaseProgressInfo
from agents.manager.default_job import DefaultJob, DefaultJobProgressInfo
from agents.manager.job_types import RunnerKind
from agents.manager.manager import (
    Manager,
    register_job_class,
    register_job_classes,
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
    "Manager",
    "register_job_class",
    "register_job_classes",
    "registered_job_classes",
    "NerfStudioJob",
    "JobRuntimeParams",
    "ProcessInfo",
]
