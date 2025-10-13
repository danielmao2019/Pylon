"""Shared enumerations and type helpers for manager jobs."""

from __future__ import annotations

from enum import Enum, unique


@unique
class RunnerKind(str, Enum):
    """Enumerates the known runner categories handled by the manager."""

    TRAINER = "trainer"
    EVALUATOR = "evaluator"
    NERFSTUDIO = "nerfstudio"
    NERFSTUDIO_GENERATION = "nerfstudio_generation"
    NERFSTUDIO_DATA = "nerfstudio_data"
    LAS_TO_PLY = "las_to_ply_offsets"
    DENSE_POINT_CLOUD = "dense_point_cloud"
    SPARSE_POINT_CLOUD = "sparse_point_cloud"
