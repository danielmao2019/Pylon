"""Data structures for LOD benchmark."""

from typing import Dict, Any
import torch
from dataclasses import dataclass


@dataclass
class PointCloudSample:
    """A single point cloud sample with metadata."""
    name: str
    points: torch.Tensor
    colors: torch.Tensor
    source: str
    metadata: Dict[str, Any] = None


@dataclass
class CameraPose:
    """A camera pose configuration."""
    camera_state: Dict[str, Any]
    distance_group: str  # 'close', 'medium', 'far'
    distance_value: float
    pose_id: int


@dataclass
class BenchmarkStats:
    """Statistics from a single benchmark run."""
    point_cloud_name: str
    camera_pose_info: str
    original_points: int
    final_points: int
    point_reduction_pct: float
    lod_level: int
    no_lod_time: float
    lod_time: float
    speedup_ratio: float
    num_runs: int