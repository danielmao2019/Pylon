"""Density-based Level of Detail system with percentage-based subsampling."""
from typing import Dict
import torch

from utils.input_checks.check_point_cloud import check_point_cloud
from utils.point_cloud_ops.random_select import RandomSelect
import logging

logger = logging.getLogger(__name__)


# Global cache that persists across function calls
_global_density_cache: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
_global_density_original_cache: Dict[str, Dict[str, torch.Tensor]] = {}


class DensityLOD:
    """Density-based Level of Detail with percentage-based subsampling and caching.
    
    This system allows users to control the percentage of points to display
    when LOD type is set to 'none'. It uses caching to avoid recomputing
    subsampled point clouds when the camera changes.
    """

    def __init__(self, seed: int = 42):
        """Initialize density LOD system.

        Args:
            seed: Random seed for reproducible subsampling (default: 42)
        """
        self.seed = seed

    def subsample(
        self,
        point_cloud: Dict[str, torch.Tensor],
        density_percentage: int,
        point_cloud_id: str
    ) -> Dict[str, torch.Tensor]:
        """Subsample point cloud to specified density percentage.

        Args:
            point_cloud: Original point cloud data
            density_percentage: Percentage of points to keep (1-100)
            point_cloud_id: String identifier for the point cloud
                           e.g., "pcr/kitti:42:source" or "change_detection:10:union"

        Returns:
            Subsampled point cloud at specified density
        """
        check_point_cloud(point_cloud)
        assert isinstance(density_percentage, int), f"density_percentage must be int, got {type(density_percentage)}"
        assert isinstance(point_cloud_id, str), f"point_cloud_id must be str, got {type(point_cloud_id)}"
        assert 1 <= density_percentage <= 100, f"density_percentage must be 1-100, got {density_percentage}"
        
        # If density is 100%, return original point cloud without caching overhead
        if density_percentage == 100:
            if point_cloud_id in _global_density_original_cache:
                return _global_density_original_cache[point_cloud_id]
            else:
                return point_cloud
        
        # Ensure we have the original point cloud cached
        if point_cloud_id not in _global_density_original_cache:
            _global_density_original_cache[point_cloud_id] = point_cloud

        # Check if this density percentage is already cached
        if (point_cloud_id in _global_density_cache and 
            density_percentage in _global_density_cache[point_cloud_id]):
            logger.info(f"Density cache hit: ID={point_cloud_id}, Density={density_percentage}%")
            return _global_density_cache[point_cloud_id][density_percentage]

        # Compute subsampled point cloud
        original_pc = _global_density_original_cache[point_cloud_id]
        subsampled_pc = self._subsample_point_cloud(original_pc, density_percentage)
        
        # Cache the result
        if point_cloud_id not in _global_density_cache:
            _global_density_cache[point_cloud_id] = {}
        _global_density_cache[point_cloud_id][density_percentage] = subsampled_pc
        
        # Log density information
        original_count = len(original_pc['pos'])
        subsampled_count = len(subsampled_pc['pos'])
        logger.info(f"Density LOD: ID={point_cloud_id}, Density={density_percentage}%, Points={subsampled_count}/{original_count}")

        return subsampled_pc

    def _subsample_point_cloud(
        self,
        point_cloud: Dict[str, torch.Tensor],
        density_percentage: int
    ) -> Dict[str, torch.Tensor]:
        """Subsample point cloud to target density percentage.

        Uses RandomSelect for consistent percentage-based sampling with
        deterministic seeding for reproducible results.
        
        Args:
            point_cloud: Original point cloud dictionary
            density_percentage: Target density percentage (1-100)
            
        Returns:
            Subsampled point cloud dictionary
        """
        check_point_cloud(point_cloud)
        
        # Convert percentage to decimal and use RandomSelect
        percentage_decimal = density_percentage / 100.0
        
        # Use RandomSelect utility with the percentage and seed (following BaseTransform API pattern)
        return RandomSelect(percentage=percentage_decimal)(point_cloud, seed=self.seed)
