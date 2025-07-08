"""Screen-space aware Level of Detail utilities for point cloud visualization.

This module implements a principled LOD system that:
1. Projects 3D points to actual 2D screen space using camera parameters
2. Calculates real point density per pixel in the rendered view
3. Only reduces points when density exceeds optimal threshold (>2-3 points/pixel)
4. Preserves all points that contribute unique visual information

Key principle: LOD should only reduce redundant points per pixel, never degrade visual quality.
"""
from typing import Dict, Tuple, Union, Optional, Any
import math
import numpy as np
import torch
from data.transforms.vision_3d.downsample import DownSample
from utils.input_checks.point_cloud import check_point_cloud


# LOD system constants
MIN_SHAPE_PRESERVATION_POINTS = 2000
MIN_REDUCTION_THRESHOLD = 0.95  # Only apply LOD if reducing by >5%
FOV_DEGREES = 60.0  # Assumed field of view for screen coverage calculation
COMPLEXITY_SCALING_FACTOR = 5.0  # Scaling factor for complexity ratio
DISTANCE_FACTOR_MIN = 0.5  # Never reduce distance factor below 50%

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
        
    def calculate_target_points(
        self,
        point_cloud: Dict[str, torch.Tensor],
        camera_state: Dict[str, Any],
        point_cloud_id: str,
        viewport_size: Tuple[int, int] = (800, 600)
    ) -> int:
        """Calculate target points based on screen-space pixel density analysis.
        
        Core principle: Only reduce points when there are redundant points per pixel.
        
        Args:
            point_cloud: Point cloud dictionary with 'pos' key
            camera_state: Camera state with eye, center, up
            point_cloud_id: Unique identifier for caching and hysteresis
            viewport_size: Screen viewport size in pixels (width, height)
            
        Returns:
            Optimal target point count that eliminates redundancy without quality loss
        """
        points = point_cloud['pos']
        total_points = points.shape[0]
        
        # Project 3D points to 2D screen coordinates
        screen_coords = self._project_to_screen(points, camera_state, viewport_size)
        
        # If no points are visible, keep minimal set for shape preservation
        if len(screen_coords) == 0:
            return max(MIN_SHAPE_PRESERVATION_POINTS, total_points // 10)
            
        # Calculate actual point density per pixel
        occupied_pixels = self._count_occupied_pixels(screen_coords, viewport_size)
        current_density = len(screen_coords) / max(occupied_pixels, 1)
        
        # Only reduce if density exceeds target (too many points per pixel)
        target_density = self.config['target_points_per_pixel']
        if current_density <= target_density:
            return total_points  # Already optimal - no reduction needed
            
        # Calculate optimal points to achieve target density
        optimal_points = int(occupied_pixels * target_density)
        
        # Apply conservative bounds
        optimal_points = max(optimal_points, MIN_SHAPE_PRESERVATION_POINTS)
        optimal_points = min(optimal_points, total_points)
        
        # Apply hysteresis to prevent flickering
        final_target = self._apply_hysteresis(point_cloud_id, optimal_points)
        
        return final_target
        
    def _project_to_screen(
        self,
        points: torch.Tensor,
        camera_state: Dict[str, Any],
        viewport_size: Tuple[int, int]
    ) -> np.ndarray:
        """Project 3D points to 2D screen coordinates using camera parameters."""
        points_np = points.cpu().numpy()
        
        # Get camera parameters
        eye = camera_state.get('eye', {'x': 1.5, 'y': 1.5, 'z': 1.5})
        center = camera_state.get('center', {'x': 0, 'y': 0, 'z': 0})
        up = camera_state.get('up', {'x': 0, 'y': 0, 'z': 1})
        
        # Convert to numpy arrays
        eye_pos = np.array([eye['x'], eye['y'], eye['z']])
        center_pos = np.array([center['x'], center['y'], center['z']])
        up_vec = np.array([up['x'], up['y'], up['z']])
        
        # Create view matrix
        forward = center_pos - eye_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up_vec)
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(right, forward)
        
        # Transform points to camera space
        points_cam = points_np - eye_pos
        x_cam = np.dot(points_cam, right)
        y_cam = np.dot(points_cam, up_corrected)
        z_cam = -np.dot(points_cam, forward)
        
        # Perspective projection
        fov_rad = math.radians(FOV_DEGREES)
        aspect_ratio = viewport_size[0] / viewport_size[1]
        
        # Avoid division by zero for points behind camera
        z_cam = np.maximum(z_cam, 1e-6)
        
        # Project to normalized device coordinates
        f = 1.0 / math.tan(fov_rad / 2.0)
        x_ndc = (x_cam * f) / (z_cam * aspect_ratio)
        y_ndc = (y_cam * f) / z_cam
        
        # Convert to screen coordinates
        x_screen = (x_ndc + 1.0) * 0.5 * viewport_size[0]
        y_screen = (1.0 - y_ndc) * 0.5 * viewport_size[1]
        
        # Filter points in screen bounds and in front of camera
        valid_mask = (
            (x_screen >= 0) & (x_screen < viewport_size[0]) &
            (y_screen >= 0) & (y_screen < viewport_size[1]) &
            (z_cam > 0)
        )
        
        return np.column_stack([x_screen[valid_mask], y_screen[valid_mask]])
        
    def _count_occupied_pixels(
        self,
        screen_coords: np.ndarray,
        viewport_size: Tuple[int, int]
    ) -> int:
        """Count number of unique pixels occupied by points."""
        if len(screen_coords) == 0:
            return 0
            
        # Round to nearest pixel coordinates
        pixel_coords = np.round(screen_coords).astype(int)
        
        # Remove duplicates to count unique pixels
        unique_pixels = np.unique(pixel_coords, axis=0)
        
        return len(unique_pixels)
        
            
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
            return 0.1  # Fallback for degenerate/flat point clouds
            
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
