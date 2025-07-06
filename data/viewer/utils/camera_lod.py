"""Camera-dependent Level of Detail utilities for point cloud visualization."""
from typing import Dict, Tuple, Union, Optional, Any
import math
import numpy as np
import torch
from data.transforms.vision_3d.downsample import DownSample
from utils.input_checks.point_cloud import check_point_cloud


class CameraLODManager:
    """Manages Level of Detail based on camera distance from point cloud."""
    
    # LOD level definitions (target point counts)
    LOD_LEVELS = {
        0: None,    # Original (no downsampling)
        1: 50000,   # High detail
        2: 25000,   # Medium detail  
        3: 10000,   # Low detail
    }
    
    # Distance thresholds for automatic LOD selection
    # These are normalized by point cloud size
    DISTANCE_THRESHOLDS = {
        0: 0.0,    # Very close: full detail
        1: 0.5,    # Close: high detail
        2: 2.0,    # Medium: medium detail  
        3: 5.0,    # Far: low detail
    }
    
    def __init__(self, hysteresis_factor: float = 0.2):
        """Initialize the LOD manager.
        
        Args:
            hysteresis_factor: Factor to prevent LOD flickering (0.0-1.0)
        """
        self.hysteresis_factor = hysteresis_factor
        self._lod_cache = {}  # Cache for downsampled point clouds
        self._current_lod = {}  # Track current LOD for each point cloud
        
    def calculate_camera_distance(
        self, 
        camera_state: Dict[str, Any], 
        point_cloud_center: np.ndarray,
        point_cloud_bounds: Tuple[float, float]
    ) -> float:
        """Calculate normalized distance from camera to point cloud.
        
        Args:
            camera_state: Plotly camera state with 'eye', 'center', 'up'
            point_cloud_center: 3D center point of the point cloud
            point_cloud_bounds: (min_extent, max_extent) of point cloud
            
        Returns:
            Normalized distance (distance / point_cloud_size)
        """
        # Extract camera eye position
        eye = camera_state.get('eye', {'x': 1.5, 'y': 1.5, 'z': 1.5})
        camera_pos = np.array([eye['x'], eye['y'], eye['z']])
        
        # Calculate euclidean distance
        distance = np.linalg.norm(camera_pos - point_cloud_center)
        
        # Normalize by point cloud size (max extent)
        pc_size = max(point_cloud_bounds[1] - point_cloud_bounds[0], 1e-6)  # Avoid division by zero
        normalized_distance = distance / pc_size
        
        return normalized_distance
        
    def get_lod_level(
        self, 
        camera_distance: float, 
        point_cloud_id: str,
        force_level: Optional[int] = None
    ) -> int:
        """Determine appropriate LOD level based on camera distance.
        
        Args:
            camera_distance: Normalized camera distance
            point_cloud_id: Unique identifier for this point cloud
            force_level: Force specific LOD level (overrides auto calculation)
            
        Returns:
            LOD level (0-3)
        """
        if force_level is not None:
            level = max(0, min(3, force_level))
            self._current_lod[point_cloud_id] = level
            return level
            
        # Get current LOD level for hysteresis calculation
        current_lod = self._current_lod.get(point_cloud_id, None)
        
        # Determine target LOD level based on distance
        target_lod = 0  # Default to highest detail
        for level in sorted(self.DISTANCE_THRESHOLDS.keys(), reverse=True):
            if camera_distance >= self.DISTANCE_THRESHOLDS[level]:
                target_lod = level
                break
                
        # Apply hysteresis to prevent flickering (only if we have a previous LOD level)
        if current_lod is not None and target_lod != current_lod:
            # Calculate hysteresis threshold
            current_threshold = self.DISTANCE_THRESHOLDS[current_lod]
            hysteresis_buffer = current_threshold * self.hysteresis_factor
            
            # Only change LOD if we're significantly past the threshold
            if target_lod > current_lod:  # Moving to lower detail
                if camera_distance > current_threshold + hysteresis_buffer:
                    self._current_lod[point_cloud_id] = target_lod
                    return target_lod
            else:  # Moving to higher detail  
                target_threshold = self.DISTANCE_THRESHOLDS[target_lod]
                if camera_distance < target_threshold - hysteresis_buffer:
                    self._current_lod[point_cloud_id] = target_lod
                    return target_lod
                    
            # Stay at current LOD due to hysteresis
            return current_lod
        
        # First time or no hysteresis needed
        self._current_lod[point_cloud_id] = target_lod
        return target_lod
        
    def get_downsampled_point_cloud(
        self,
        point_cloud: Dict[str, torch.Tensor], 
        lod_level: int,
        point_cloud_id: str
    ) -> Dict[str, torch.Tensor]:
        """Get downsampled version of point cloud for given LOD level.
        
        Args:
            point_cloud: Original point cloud dictionary
            lod_level: Target LOD level (0-3)
            point_cloud_id: Unique identifier for caching
            
        Returns:
            Downsampled point cloud (or original if LOD 0)
        """
        check_point_cloud(point_cloud)
        
        # LOD 0 returns original point cloud
        if lod_level == 0:
            return point_cloud
            
        # Check cache first
        cache_key = f"{point_cloud_id}_lod_{lod_level}"
        if cache_key in self._lod_cache:
            return self._lod_cache[cache_key]
            
        # Get target point count for this LOD level
        target_points = self.LOD_LEVELS[lod_level]
        current_points = point_cloud['pos'].shape[0]
        
        # If already below target, return original
        if current_points <= target_points:
            self._lod_cache[cache_key] = point_cloud
            return point_cloud
            
        # Calculate voxel size for downsampling
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
        """Calculate voxel size to achieve approximately target point count.
        
        Args:
            point_cloud: Input point cloud
            target_points: Desired number of points after downsampling
            
        Returns:
            Voxel size for downsampling
        """
        points = point_cloud['pos']
        
        # Calculate bounding box
        min_coords = points.min(dim=0)[0]
        max_coords = points.max(dim=0)[0]
        bbox_size = max_coords - min_coords
        
        # Estimate volume and density
        volume = torch.prod(bbox_size).item()
        current_density = points.shape[0] / volume
        target_density = target_points / volume
        
        # Calculate voxel size based on density ratio
        density_ratio = target_density / current_density
        voxel_size = 1.0 / math.sqrt(density_ratio)
        
        # Apply bounds to prevent extreme voxel sizes
        max_bbox_dim = torch.max(bbox_size).item()
        min_voxel = max_bbox_dim / 1000  # At least 1000 voxels per dimension
        max_voxel = max_bbox_dim / 10    # At most 10 voxels per dimension
        
        voxel_size = max(min_voxel, min(max_voxel, voxel_size))
        
        return voxel_size
        
    def clear_cache(self, point_cloud_id: Optional[str] = None):
        """Clear LOD cache for memory management.
        
        Args:
            point_cloud_id: Clear cache for specific point cloud, or all if None
        """
        if point_cloud_id is None:
            self._lod_cache.clear()
            self._current_lod.clear()
        else:
            # Clear cache entries for this point cloud
            keys_to_remove = [k for k in self._lod_cache.keys() if k.startswith(f"{point_cloud_id}_")]
            for key in keys_to_remove:
                del self._lod_cache[key]
            if point_cloud_id in self._current_lod:
                del self._current_lod[point_cloud_id]
                
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring.
        
        Returns:
            Dictionary with cache size and current LOD levels
        """
        return {
            'cached_lods': len(self._lod_cache),
            'tracked_point_clouds': len(self._current_lod),
            'current_lods': dict(self._current_lod)
        }


def calculate_point_cloud_bounds(points: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Calculate point cloud center and bounds for LOD calculations.
    
    Args:
        points: Point cloud positions (N, 3)
        
    Returns:
        Tuple of (center, (min_extent, max_extent))
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        
    center = points.mean(axis=0)
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    return center, (min_coords.min(), max_coords.max())


# Global LOD manager instance
_lod_manager = CameraLODManager()


def get_lod_manager() -> CameraLODManager:
    """Get the global LOD manager instance."""
    return _lod_manager
