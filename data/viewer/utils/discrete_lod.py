"""Discrete Level of Detail system with pre-computed levels."""
from typing import Dict, Optional, Any
import torch
from utils.input_checks.point_cloud import check_point_cloud
from data.viewer.utils.lod_utils import get_camera_position


class DiscreteLOD:
    """Discrete Level of Detail with pre-computed downsampling levels."""
    
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
        
    def get_lod_levels(self, point_cloud: Dict[str, torch.Tensor]) -> Dict[int, int]:
        """Get point counts for each LOD level.
        
        Returns:
            Dictionary mapping level -> point count
        """
        total_points = point_cloud['pos'].shape[0]
        levels = {}
        
        for level in range(self.num_levels + 1):
            level_points = int(total_points * (self.reduction_factor ** level))
            levels[level] = max(level_points, 1000)  # Minimum points
            
        return levels
        
    def has_levels(self, point_cloud_id: str) -> bool:
        """Check if LOD levels have been pre-computed for this point cloud."""
        return point_cloud_id in self._lod_cache
        
    def precompute_levels(
        self, 
        point_cloud: Dict[str, torch.Tensor], 
        point_cloud_id: str
    ) -> None:
        """Pre-compute all LOD levels for a point cloud."""
        check_point_cloud(point_cloud)
        
        if point_cloud_id in self._lod_cache:
            return  # Already computed
            
        levels = {}
        levels[0] = point_cloud  # Level 0 = original
        
        current_pc = point_cloud
        for level in range(1, self.num_levels + 1):
            target_points = int(current_pc['pos'].shape[0] * self.reduction_factor)
            target_points = max(target_points, 1000)
            
            # Use voxel grid downsampling for pre-computation
            downsampled_pc = self._voxel_downsample(current_pc, target_points)
            levels[level] = downsampled_pc
            current_pc = downsampled_pc
            
        self._lod_cache[point_cloud_id] = levels
        
    def select_level(
        self,
        point_cloud_id: str,
        camera_state: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Select appropriate LOD level based on camera distance."""
        if point_cloud_id not in self._lod_cache:
            raise ValueError(f"Point cloud {point_cloud_id} not pre-computed. Call precompute_levels first.")
            
        levels = self._lod_cache[point_cloud_id]
        return self._select_by_distance(levels, camera_state)
            
    def _select_by_distance(self, levels: Dict[int, Dict[str, torch.Tensor]], camera_state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Select level based on camera distance."""
        original_pc = levels[0]
        points = original_pc['pos']
        
        # Get camera position on same device as points
        camera_pos = get_camera_position(camera_state, device=points.device, dtype=points.dtype)
        
        # Calculate average distance
        distances = torch.norm(points - camera_pos, dim=1)
        avg_distance = distances.mean().item()
        
        # Distance-based level selection using configurable thresholds
        thresholds = self.distance_thresholds
        if avg_distance < thresholds['close']:
            level = 0  # Close: use original
        elif avg_distance < thresholds['medium_close']:
            level = 1  # Medium close
        elif avg_distance < thresholds['medium_far']:
            level = 2  # Medium far
        else:
            level = min(3, self.num_levels)  # Far: aggressive reduction
            
        return levels[level]
        
    def _voxel_downsample(
        self, 
        point_cloud: Dict[str, torch.Tensor], 
        target_points: int
    ) -> Dict[str, torch.Tensor]:
        """Simple voxel grid downsampling for pre-computation."""
        points = point_cloud['pos']
        
        # Calculate voxel size to achieve approximately target points
        bbox_size = points.max(dim=0)[0] - points.min(dim=0)[0]
        volume = torch.prod(bbox_size).item()
        
        if volume <= 0:
            # Degenerate case - random sampling
            indices = torch.randperm(points.shape[0])[:target_points]
            return {key: tensor[indices] for key, tensor in point_cloud.items()}
            
        # Estimate voxel size
        current_density = points.shape[0] / volume
        target_density = target_points / volume
        voxel_size = (current_density / target_density) ** (1/3) if target_density > 0 else 1.0
        
        # Apply voxel grid
        voxel_coords = ((points - points.min(dim=0)[0]) / voxel_size).long()
        
        # Find unique voxels and select first point from each
        # Use large multipliers to avoid hash collisions
        hash_multiplier_x = 1000000
        hash_multiplier_y = 1000
        voxel_hash = (
            voxel_coords[:, 0] * hash_multiplier_x + 
            voxel_coords[:, 1] * hash_multiplier_y + 
            voxel_coords[:, 2]
        )
        
        unique_voxels, unique_indices = torch.unique(voxel_hash, return_inverse=True)
        
        # Select one point per voxel
        selected_indices = []
        for i, voxel in enumerate(unique_voxels):
            voxel_points = torch.nonzero(unique_indices == i, as_tuple=True)[0]
            selected_indices.append(voxel_points[0].item())  # Take first point
            
        selected_indices = torch.tensor(selected_indices, device=points.device, dtype=torch.long)
        
        # Add random points if needed to reach target
        selected_indices = self._fill_to_target(selected_indices, target_points, points.shape[0])
                
        return {key: tensor[selected_indices] for key, tensor in point_cloud.items()}
        
    def _fill_to_target(self, selected_indices: torch.Tensor, target_points: int, total_points: int) -> torch.Tensor:
        """Fill selected indices to reach target count with random points."""
        if len(selected_indices) >= target_points:
            return selected_indices
            
        device = selected_indices.device
        remaining = target_points - len(selected_indices)
        all_indices = torch.arange(total_points, device=device)
        available_mask = torch.ones(total_points, dtype=torch.bool, device=device)
        available_mask[selected_indices] = False
        available_indices = all_indices[available_mask]
        
        if len(available_indices) > 0:
            additional_count = min(remaining, len(available_indices))
            additional = available_indices[torch.randperm(len(available_indices))[:additional_count]]
            selected_indices = torch.cat([selected_indices, additional])
            
        return selected_indices
        
    def clear_cache(self, point_cloud_id: Optional[str] = None):
        """Clear pre-computed LOD levels."""
        if point_cloud_id is None:
            self._lod_cache.clear()
        elif point_cloud_id in self._lod_cache:
            del self._lod_cache[point_cloud_id]