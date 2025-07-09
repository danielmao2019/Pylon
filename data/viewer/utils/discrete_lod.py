"""Discrete Level of Detail system with pre-computed levels."""
from typing import Any, Dict, Optional
import torch
from utils.input_checks.point_cloud import check_point_cloud
from utils.point_cloud_ops.random_select import RandomSelect
from data.viewer.utils.lod_utils import get_camera_position


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
            'close': 2.0,
            'medium_close': 5.0,
            'medium_far': 10.0
        }
        self._lod_cache: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
        self._original_point_clouds: Dict[str, Dict[str, torch.Tensor]] = {}

    def subsample(
        self,
        point_cloud_id: str,
        camera_state: Dict[str, Any],
        point_cloud: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Subsample point cloud based on camera distance.

        Args:
            point_cloud_id: Unique identifier for the point cloud
            camera_state: Camera position and orientation information
            point_cloud: Original point cloud data (required if not cached)

        Returns:
            Subsampled point cloud at appropriate LOD level
        """
        # Ensure we have the original point cloud
        if point_cloud_id not in self._original_point_clouds:
            if point_cloud is None:
                raise ValueError(f"Point cloud {point_cloud_id} not found. Provide point_cloud parameter.")
            self._original_point_clouds[point_cloud_id] = point_cloud

        # Compute LOD levels if not already done
        if point_cloud_id not in self._lod_cache:
            original_pc = self._original_point_clouds[point_cloud_id]
            self._precompute_lod_levels(point_cloud_id, original_pc)

        # Determine target level based on camera distance
        target_level = self._determine_target_level(point_cloud_id, camera_state)

        # Return precomputed subsampled point cloud
        return self._lod_cache[point_cloud_id][target_level]

    def clear_cache(self, point_cloud_id: Optional[str] = None) -> None:
        """Clear cached LOD levels and original point clouds."""
        if point_cloud_id is None:
            self._lod_cache.clear()
            self._original_point_clouds.clear()
        elif point_cloud_id in self._lod_cache:
            del self._lod_cache[point_cloud_id]
            if point_cloud_id in self._original_point_clouds:
                del self._original_point_clouds[point_cloud_id]

    def _precompute_lod_levels(
        self,
        point_cloud_id: str,
        point_cloud: Dict[str, torch.Tensor]
    ) -> None:
        """Pre-compute all LOD levels for a point cloud."""
        check_point_cloud(point_cloud)

        levels = {}
        levels[0] = point_cloud  # Level 0 = original

        current_pc = point_cloud
        for level in range(1, self.num_levels + 1):
            target_points = int(current_pc['pos'].shape[0] * self.reduction_factor)
            target_points = max(target_points, self.MIN_POINTS_PER_LEVEL)

            downsampled_pc = self._downsample_point_cloud(current_pc, target_points)
            levels[level] = downsampled_pc
            current_pc = downsampled_pc

        self._lod_cache[point_cloud_id] = levels

    def _downsample_point_cloud(
        self,
        point_cloud: Dict[str, torch.Tensor],
        target_points: int
    ) -> Dict[str, torch.Tensor]:
        """Downsample point cloud to target number of points.

        Uses RandomSelect for clean percentage-based sampling that ensures
        discrete LOD levels have predictable point count reductions.
        """
        current_count = len(point_cloud['pos'])

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
        original_pc = self._original_point_clouds[point_cloud_id]
        points = original_pc['pos']

        # Get camera position on same device as points
        camera_pos = get_camera_position(camera_state, device=points.device, dtype=points.dtype)

        # Calculate point cloud bounding box diagonal
        min_coords = points.min(dim=0)[0]
        max_coords = points.max(dim=0)[0]
        diagonal_size = torch.norm(max_coords - min_coords).item()

        # Calculate average distance relative to diagonal size
        distances = torch.norm(points - camera_pos, dim=1)
        avg_distance = distances.mean().item()
        relative_distance = avg_distance / diagonal_size if diagonal_size > 0 else 0.0

        # Distance-based level selection using relative thresholds
        thresholds = self.distance_thresholds
        if relative_distance < thresholds['close']:
            return 0  # Close: use original
        elif relative_distance < thresholds['medium_close']:
            return 1  # Medium close
        elif relative_distance < thresholds['medium_far']:
            return 2  # Medium far
        else:
            return min(3, self.num_levels)  # Far: aggressive reduction
