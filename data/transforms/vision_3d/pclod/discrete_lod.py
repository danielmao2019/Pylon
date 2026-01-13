"""Discrete Level of Detail system with pre-computed levels."""
from typing import Any, Dict, Optional
import torch

from data.transforms.vision_3d.pclod.lod_utils import get_camera_position
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.random_select import RandomSelect
import logging

logger = logging.getLogger(__name__)


# Global cache that persists across function calls
_global_lod_cache: Dict[str, Dict[int, PointCloud]] = {}
_global_original_cache: Dict[str, PointCloud] = {}
_global_geometry_cache: Dict[str, Dict[str, Any]] = {}  # Cache for bounding box, center, diagonal


class DiscreteLOD:
    """Discrete Level of Detail with pre-computed downsampling levels."""

    # Class constants
    MIN_POINTS_PER_LEVEL = 1000

    def __init__(
        self,
        reduction_factor: float = 0.5,
        num_levels: int = 4,
        distance_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize discrete LOD system.

        Args:
            reduction_factor: Point reduction per level (default: 0.5)
            num_levels: Number of LOD levels to pre-compute (default: 4)
            distance_thresholds: Distance ranges for level selection (default: auto)
        """
        self.reduction_factor = reduction_factor
        self.num_levels = num_levels
        self.distance_thresholds = distance_thresholds or {
            'close': 0.02,      # Very close - use original (Level 0)
            'medium_close': 0.04, # Close - light reduction (Level 1)
            'medium_far': 0.08    # Medium - more reduction (Level 2)
        }

    def subsample(
        self,
        point_cloud: PointCloud,
        camera_state: Dict[str, Any],
        point_cloud_id: str
    ) -> PointCloud:
        """Subsample point cloud based on camera distance.

        Args:
            point_cloud: Original point cloud data
            camera_state: Camera position and orientation information
            point_cloud_id: String identifier for the point cloud
                           e.g., "pcr/kitti:42:source" or "change_detection:10:union"

        Returns:
            Subsampled point cloud at appropriate LOD level
        """
        assert isinstance(point_cloud, PointCloud), f"{type(point_cloud)=}"
        assert isinstance(point_cloud_id, str), f"point_cloud_id must be str, got {type(point_cloud_id)}"

        # Ensure we have the original point cloud cached
        if point_cloud_id not in _global_original_cache:
            _global_original_cache[point_cloud_id] = point_cloud

        # Compute LOD levels if not already done
        if point_cloud_id not in _global_lod_cache:
            original_pc = _global_original_cache[point_cloud_id]
            self._precompute_lod_levels(point_cloud_id, original_pc)

        # Cache geometry properties if not already done
        if point_cloud_id not in _global_geometry_cache:
            self._precompute_geometry_properties(point_cloud_id)

        # Determine target level based on camera distance
        target_level = self._determine_target_level(point_cloud_id, camera_state)

        # Log LOD information
        original_count = _global_original_cache[point_cloud_id].num_points
        subsampled_count = _global_lod_cache[point_cloud_id][target_level].num_points
        logger.info(f"Discrete LOD: ID={point_cloud_id}, Level={target_level}, Points={subsampled_count}/{original_count}")

        # Return precomputed subsampled point cloud
        return _global_lod_cache[point_cloud_id][target_level]

    def _precompute_lod_levels(
        self,
        point_cloud_id: str,
        point_cloud: PointCloud
    ) -> None:
        """Pre-compute all LOD levels for a point cloud."""
        levels: Dict[int, PointCloud] = {}
        levels[0] = point_cloud  # Level 0 = original

        current_pc = point_cloud
        for level in range(1, self.num_levels + 1):
            target_points = int(current_pc.xyz.shape[0] * self.reduction_factor)
            target_points = max(target_points, self.MIN_POINTS_PER_LEVEL)

            downsampled_pc = self._downsample_point_cloud(current_pc, target_points)
            levels[level] = downsampled_pc
            current_pc = downsampled_pc

        _global_lod_cache[point_cloud_id] = levels

    def _precompute_geometry_properties(self, point_cloud_id: str) -> None:
        """Pre-compute and cache geometry properties for faster distance calculations."""
        original_pc = _global_original_cache[point_cloud_id]
        points = original_pc.xyz

        # Calculate bounding box once and cache it
        min_coords = points.min(dim=0)[0]
        max_coords = points.max(dim=0)[0]
        center_point = (min_coords + max_coords) / 2
        diagonal_size = torch.norm(max_coords - min_coords).item()

        # Cache all geometry properties
        _global_geometry_cache[point_cloud_id] = {
            'min_coords': min_coords,
            'max_coords': max_coords,
            'center_point': center_point,
            'diagonal_size': diagonal_size,
            'device': points.device,
            'dtype': points.dtype
        }

    def _downsample_point_cloud(
        self,
        point_cloud: PointCloud,
        target_points: int
    ) -> PointCloud:
        """Downsample point cloud to target number of points.

        Uses RandomSelect for clean percentage-based sampling that ensures
        discrete LOD levels have predictable point count reductions.
        """
        current_count = point_cloud.num_points

        if target_points >= current_count:
            return point_cloud

        # Calculate sampling percentage and use RandomSelect
        percentage = target_points / current_count
        return RandomSelect(percentage)(point_cloud)

    def _determine_target_level(
        self,
        point_cloud_id: str,
        camera_state: Dict[str, Any]
    ) -> int:
        """Determine appropriate LOD level based on camera distance."""
        # Get cached geometry properties (no expensive recomputation)
        geometry = _global_geometry_cache[point_cloud_id]
        center_point = geometry['center_point']
        diagonal_size = geometry['diagonal_size']
        device = geometry['device']
        dtype = geometry['dtype']

        # Get camera position on same device as points (works on both CPU and CUDA)
        camera_pos = get_camera_position(camera_state, device=device, dtype=dtype)

        # Calculate distance to center point relative to diagonal size (fast O(1) operation)
        center_distance = torch.norm(camera_pos - center_point).item()
        relative_distance = center_distance / diagonal_size if diagonal_size > 0 else 0.0

        # Log distance calculation details
        logger.info(f"Distance calculation: center_distance={center_distance:.2f}, diagonal={diagonal_size:.2f}, relative={relative_distance:.2f}")

        # Distance-based level selection using relative thresholds
        thresholds = self.distance_thresholds
        if relative_distance < thresholds['close']:
            level = 0  # Close: use original
        elif relative_distance < thresholds['medium_close']:
            level = 1  # Medium close
        elif relative_distance < thresholds['medium_far']:
            level = 2  # Medium far
        else:
            level = min(3, self.num_levels)  # Far: aggressive reduction

        logger.info(f"Selected LOD level: {level} (thresholds: {thresholds})")
        return level
