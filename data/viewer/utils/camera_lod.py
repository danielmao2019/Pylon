"""Camera-dependent Level of Detail utilities for point cloud visualization.

This module implements an intelligent LOD system that dynamically determines
optimal point cloud reduction based on:
1. Camera distance and viewing angle
2. Point cloud characteristics (size, density, complexity)
3. Screen space coverage and pixel density
4. Visual quality preservation constraints
"""
from typing import Dict, Tuple, Union, Optional, Any
import math
import numpy as np
import torch
from data.transforms.vision_3d.downsample import DownSample
from utils.input_checks.point_cloud import check_point_cloud


# Default LOD configuration
DEFAULT_LOD_CONFIG = {
    'target_points_per_pixel': 2.0,
    'min_quality_ratio': 0.2,  # Conservative: never less than 20%
    'max_reduction_ratio': 0.8,  # Conservative: max 80% reduction
    'hysteresis_factor': 0.15,
    'distance_scaling_factor': 0.3,
    'complexity_base_factor': 0.8,
    'complexity_scaling': 0.2,
    'size_small_threshold': 10000,
    'size_medium_threshold': 50000,
    'size_large_threshold': 200000,
}


class LODManager:
    """Level of Detail manager that dynamically calculates optimal point reduction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LOD manager.
        
        Args:
            config: LOD configuration parameters dictionary
        """
        self.config = {**DEFAULT_LOD_CONFIG, **(config or {})}
        
        # Cache for downsampled point clouds and LOD decisions
        self._lod_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._current_target_points: Dict[str, int] = {}  # Track current target for hysteresis
        self._complexity_cache: Dict[int, Tuple[float, float]] = {}  # Cache expensive calculations
        
    def calculate_target_points(
        self,
        point_cloud: Dict[str, torch.Tensor],
        camera_state: Dict[str, Any],
        point_cloud_id: str,
        viewport_size: Tuple[int, int] = (800, 600)
    ) -> int:
        """Calculate optimal target point count based on viewing factors.
        
        Args:
            point_cloud: Point cloud dictionary with 'pos' key
            camera_state: Camera state with eye, center, up
            point_cloud_id: Unique identifier for caching and hysteresis
            viewport_size: Screen viewport size in pixels (width, height)
            
        Returns:
            Optimal target point count for current viewing conditions
        """
        total_points = point_cloud['pos'].shape[0]
        
        # Factor 1: Screen coverage analysis
        screen_target = self._calculate_screen_coverage_target(
            point_cloud, camera_state, viewport_size
        )
        
        # Factor 2: Distance-based quality scaling
        distance_factor = self._calculate_distance_quality_factor(
            point_cloud, camera_state
        )
        
        # Factor 3: Point cloud complexity preservation
        complexity_factor = self._calculate_complexity_factor(point_cloud)
        
        # Factor 4: Size-adaptive scaling
        size_factor = self._calculate_size_adaptive_factor(total_points)
        
        # Combine all factors with bounds checking
        target = int(
            screen_target * distance_factor * complexity_factor * size_factor
        )
        
        # Apply conservative quality constraints
        min_points = max(
            int(total_points * self.config['min_quality_ratio']),
            2000  # Absolute minimum for shape preservation
        )
        max_points = int(total_points * (1.0 - self.config['max_reduction_ratio']))
        
        # Clamp to constraints
        constrained_target = max(min_points, min(max_points, target))
        constrained_target = min(constrained_target, total_points)
        
        # Apply hysteresis to prevent flickering
        final_target = self._apply_hysteresis(
            point_cloud_id, constrained_target, total_points
        )
        
        return final_target
        
    def _calculate_screen_coverage_target(
        self,
        point_cloud: Dict[str, torch.Tensor],
        camera_state: Dict[str, Any],
        viewport_size: Tuple[int, int]
    ) -> int:
        """Calculate target points based on screen space coverage."""
        points = point_cloud['pos']
        
        # Calculate point cloud bounding box
        min_coords = points.min(dim=0)[0].cpu().numpy()
        max_coords = points.max(dim=0)[0].cpu().numpy()
        center = (min_coords + max_coords) / 2
        
        # Get camera parameters
        eye = camera_state.get('eye', {'x': 1.5, 'y': 1.5, 'z': 1.5})
        camera_pos = np.array([eye['x'], eye['y'], eye['z']])
        
        # Estimate screen space coverage using simple projection
        camera_distance = np.linalg.norm(camera_pos - center)
        bbox_size = np.linalg.norm(max_coords - min_coords)
        
        # Estimate screen coverage as fraction of viewport
        fov_factor = 60.0  # Assume 60 degree field of view
        angular_size = 2 * math.atan(bbox_size / (2 * max(camera_distance, 1e-6)))
        screen_coverage_ratio = angular_size / math.radians(fov_factor)
        screen_coverage_ratio = min(1.0, screen_coverage_ratio)  # Cap at 100%
        
        # Calculate screen pixels covered
        screen_pixels = viewport_size[0] * viewport_size[1] * screen_coverage_ratio
        
        # Target points per pixel
        target_points = int(screen_pixels * self.config['target_points_per_pixel'])
        
        return max(target_points, 2000)  # Minimum threshold
        
    def _calculate_distance_quality_factor(
        self,
        point_cloud: Dict[str, torch.Tensor],
        camera_state: Dict[str, Any]
    ) -> float:
        """Calculate quality factor based on camera distance."""
        points = point_cloud['pos']
        center = points.mean(dim=0).cpu().numpy()
        bbox_size = (points.max(dim=0)[0] - points.min(dim=0)[0]).norm().item()
        
        eye = camera_state.get('eye', {'x': 1.5, 'y': 1.5, 'z': 1.5})
        camera_pos = np.array([eye['x'], eye['y'], eye['z']])
        
        # Normalized distance (distance relative to object size)
        distance = np.linalg.norm(camera_pos - center)
        normalized_distance = distance / max(bbox_size, 1e-6)
        
        # Modified inverse relationship: closer = more detail needed
        # Conservative approach: don't reduce too aggressively
        distance_factor = 1.0 / (1.0 + normalized_distance * self.config['distance_scaling_factor'])
        
        # Clamp to conservative range
        return max(0.5, min(1.0, distance_factor))  # Never less than 50%
        
    def _calculate_complexity_factor(self, point_cloud: Dict[str, torch.Tensor]) -> float:
        """Calculate factor based on point cloud geometric complexity with caching."""
        points = point_cloud['pos']
        
        # Use cache for expensive std/range calculations
        cache_key = id(points)
        if cache_key not in self._complexity_cache:
            coord_std = points.std(dim=0).mean().item()
            coord_range = (points.max(dim=0)[0] - points.min(dim=0)[0]).mean().item()
            self._complexity_cache[cache_key] = (coord_std, coord_range)
        
        coord_std, coord_range = self._complexity_cache[cache_key]
        complexity_ratio = coord_std / max(coord_range, 1e-6)
        
        # Convert to factor: more complex = preserve more points
        complexity_factor = (
            self.config['complexity_base_factor'] + 
            self.config['complexity_scaling'] * min(1.0, complexity_ratio * 5.0)
        )
        
        return complexity_factor
        
    def _calculate_size_adaptive_factor(self, total_points: int) -> float:
        """Calculate factor based on original point cloud size."""
        if total_points < self.config['size_small_threshold']:
            return 1.0  # Small clouds: preserve most points
        elif total_points < self.config['size_medium_threshold']:
            return 0.9  # Medium clouds: moderate reduction
        elif total_points < self.config['size_large_threshold']:
            return 0.8  # Large clouds: can reduce more
        else:
            return 0.7  # Very large clouds: more aggressive reduction possible
            
    def _apply_hysteresis(
        self,
        point_cloud_id: str,
        new_target: int
    ) -> int:
        """Apply hysteresis to prevent LOD flickering during camera movement."""
        current_target = self._current_target_points.get(point_cloud_id, new_target)
        
        # Calculate relative change
        if current_target > 0:
            relative_change = abs(new_target - current_target) / current_target
            
            # Only change if relative change is significant
            if relative_change < self.config['hysteresis_factor']:
                return current_target
                
        # Update and return new target
        self._current_target_points[point_cloud_id] = new_target
        return new_target
        
    def get_downsampled_point_cloud(
        self,
        point_cloud: Dict[str, torch.Tensor],
        target_points: int,
        point_cloud_id: str
    ) -> Dict[str, torch.Tensor]:
        """Get downsampled point cloud with target point count."""
        check_point_cloud(point_cloud)
        
        current_points = point_cloud['pos'].shape[0]
        
        # If target is same or larger than current, return original
        if target_points >= current_points:
            return point_cloud
            
        # Check cache with optimized key generation
        cache_key = f"{point_cloud_id}_{target_points}"
        if cache_key in self._lod_cache:
            return self._lod_cache[cache_key]
            
        # Calculate voxel size for target point count
        voxel_size = self._calculate_voxel_size(point_cloud, target_points)
        
        # Perform downsampling
        downsampler = DownSample(voxel_size)
        downsampled_pc = downsampler(point_cloud)
        
        # Cache the result
        self._lod_cache[cache_key] = downsampled_pc
        
        return downsampled_pc
        
    def _calculate_voxel_size(
        self,
        point_cloud: Dict[str, torch.Tensor],
        target_points: int
    ) -> float:
        """Calculate optimal voxel size to achieve target point count."""
        points = point_cloud['pos']
        
        # Calculate bounding box
        min_coords = points.min(dim=0)[0]
        max_coords = points.max(dim=0)[0]
        bbox_size = max_coords - min_coords
        
        # Estimate volume and density
        volume = torch.prod(bbox_size).item()
        if volume <= 0:
            return 0.1  # Fallback for degenerate cases
            
        current_density = points.shape[0] / volume
        target_density = target_points / volume
        
        # Calculate voxel size based on density ratio
        if current_density > 0:
            density_ratio = target_density / current_density
            voxel_size = 1.0 / math.pow(max(density_ratio, 1e-6), 1/3)  # Cube root for 3D
        else:
            voxel_size = 1.0
            
        # Apply reasonable bounds
        max_bbox_dim = torch.max(bbox_size).item()
        min_voxel = max_bbox_dim / 500   # At least 500 voxels per dimension
        max_voxel = max_bbox_dim / 5     # At most 5 voxels per dimension
        
        voxel_size = max(min_voxel, min(max_voxel, voxel_size))
        
        return voxel_size
        
    def clear_cache(self, point_cloud_id: Optional[str] = None):
        """Clear cache for memory management."""
        if point_cloud_id is None:
            self._lod_cache.clear()
            self._current_target_points.clear()
            self._complexity_cache.clear()
        else:
            # Clear cache entries for specific point cloud
            keys_to_remove = [k for k in self._lod_cache.keys() if point_cloud_id in k]
            for key in keys_to_remove:
                del self._lod_cache[key]
            if point_cloud_id in self._current_target_points:
                del self._current_target_points[point_cloud_id]
                
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and current LOD decisions."""
        return {
            'cached_lods': len(self._lod_cache),
            'tracked_point_clouds': len(self._current_target_points),
            'complexity_cache_size': len(self._complexity_cache),
            'current_targets': dict(self._current_target_points)
        }


def calculate_point_cloud_bounds(points: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, float]:
    """Calculate point cloud center and spatial size for LOD calculations."""
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        
    center = points.mean(axis=0)
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # Calculate the diagonal distance of the bounding box
    bbox_extents = max_coords - min_coords
    diagonal_size = np.sqrt(np.sum(bbox_extents**2))
    
    return center, diagonal_size


# Global LOD manager instance for proper caching
_global_lod_manager = LODManager()


def get_lod_manager() -> LODManager:
    """Get the global LOD manager instance to ensure caching works."""
    return _global_lod_manager
