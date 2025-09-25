"""Shared enumerations and type helpers for manager jobs."""

from __future__ import annotations

from enum import Enum, unique


@unique
class RunnerKind(str, Enum):
    """Enumerates the known runner categories handled by the manager."""

    TRAINER = "trainer"
    EVALUATOR = "evaluator"
