"""Continuous Level of Detail system for point cloud visualization."""
from typing import Dict, Any
import torch
from utils.input_checks.point_cloud import check_point_cloud
from data.viewer.utils.lod_utils import get_camera_position


class ContinuousLOD:
    """Continuous Level of Detail using camera frustum binning and distance weighting."""
    
    def __init__(
        self,
        spatial_bins: int = 64,
        near_distance: float = 5.0,
        far_distance: float = 50.0,
        near_sampling_rate: float = 0.9,
        far_sampling_rate: float = 0.1
    ):
        """Initialize continuous LOD system.
        
        Args:
            spatial_bins: Number of spatial bins for coverage (default: 64)
            near_distance: Distance considered "near" to camera (default: 5.0)
            far_distance: Distance considered "far" from camera (default: 50.0)
            near_sampling_rate: Sampling rate for near bins (default: 0.9 = keep 90%)
            far_sampling_rate: Sampling rate for far bins (default: 0.1 = keep 10%)
        """
        self.spatial_bins = spatial_bins
        self.near_distance = near_distance
        self.far_distance = far_distance
        self.near_sampling_rate = near_sampling_rate
        self.far_sampling_rate = far_sampling_rate
        
    def subsample(
        self,
        point_cloud: Dict[str, torch.Tensor],
        camera_state: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Subsample point cloud using continuous LOD approach.
        
        Each spatial bin gets a sampling rate based on its average distance from camera.
        Closer bins keep more points, farther bins keep fewer points.
        
        Args:
            point_cloud: Dictionary with 'pos' key and optional 'rgb', 'labels'
            camera_state: Camera state with 'eye', 'center', 'up'
            
        Returns:
            Subsampled point cloud dictionary
        """
        check_point_cloud(point_cloud)
        
        points = point_cloud['pos']
        
        # Get camera position
        camera_pos = get_camera_position(camera_state).to(points.device)
        
        # Calculate distances for all points
        distances = torch.norm(points - camera_pos, dim=1)
        
        # Perform spatial binning
        bin_indices = self._spatial_binning(points, camera_state)
        
        # Sample from bins based on distance
        selected_indices = self._distance_based_bin_sampling(
            distances, bin_indices
        )
        
        # Return subsampled point cloud
        return {key: tensor[selected_indices] for key, tensor in point_cloud.items()}

    def _calculate_sampling_rate(self, distance: float) -> float:
        """Calculate sampling rate based on distance from camera.
        
        Interpolates linearly between near_sampling_rate and far_sampling_rate.
        """
        if distance <= self.near_distance:
            return self.near_sampling_rate
        elif distance >= self.far_distance:
            return self.far_sampling_rate
        else:
            # Linear interpolation
            t = (distance - self.near_distance) / (self.far_distance - self.near_distance)
            return self.near_sampling_rate + t * (self.far_sampling_rate - self.near_sampling_rate)
        
    def _spatial_binning(
        self, 
        points: torch.Tensor, 
        camera_state: Dict[str, Any]
    ) -> torch.Tensor:
        """Assign points to spatial bins based on camera-relative coordinates."""
        device = points.device
        
        # Get camera vectors
        camera_pos = get_camera_position(camera_state).to(device)
        center = camera_state.get('center', {'x': 0, 'y': 0, 'z': 0})
        up = camera_state.get('up', {'x': 0, 'y': 0, 'z': 1})
        
        center_pos = torch.tensor([center['x'], center['y'], center['z']], device=device, dtype=points.dtype)
        up_vec = torch.tensor([up['x'], up['y'], up['z']], device=device, dtype=points.dtype)
        
        # Create camera-aligned coordinate system
        forward = center_pos - camera_pos
        forward = forward / torch.norm(forward)
        right = torch.cross(forward, up_vec)
        right = right / torch.norm(right)
        up_corrected = torch.cross(right, forward)
        
        # Transform points to camera space
        points_cam = points - camera_pos.unsqueeze(0)
        x_cam = torch.sum(points_cam * right.unsqueeze(0), dim=1)
        y_cam = torch.sum(points_cam * up_corrected.unsqueeze(0), dim=1)
        z_cam = torch.sum(points_cam * forward.unsqueeze(0), dim=1)
        
        # Calculate bins per dimension (approximately cube root for 3D distribution)
        bins_per_dim = max(1, int(round(self.spatial_bins ** (1/3))))
        
        # Normalize coordinates to [0, bins_per_dim)
        coords_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)
        min_coords = coords_cam.min(dim=0)[0]
        max_coords = coords_cam.max(dim=0)[0]
        
        # Handle edge case where all points are in a plane
        coord_range = max_coords - min_coords
        coord_range = torch.clamp(coord_range, min=1e-6)
        
        normalized_coords = (coords_cam - min_coords) / coord_range
        bin_coords = torch.clamp(
            (normalized_coords * bins_per_dim).long(),
            0, bins_per_dim - 1
        )
        
        # Convert 3D bin coordinates to 1D bin indices
        bin_indices = (
            bin_coords[:, 0] * bins_per_dim * bins_per_dim +
            bin_coords[:, 1] * bins_per_dim +
            bin_coords[:, 2]
        )
        
        return bin_indices
        
    def _distance_based_bin_sampling(
        self,
        distances: torch.Tensor,
        bin_indices: torch.Tensor
    ) -> torch.Tensor:
        """Sample points from bins based on distance-based sampling rates.
        
        Each bin's sampling rate is determined by its average distance from camera.
        """
        device = distances.device
        selected_indices = []
        
        unique_bins = torch.unique(bin_indices)
        
        for bin_id in unique_bins:
            bin_mask = (bin_indices == bin_id)
            if bin_mask.sum() == 0:
                continue
                
            # Get indices and distances for this bin
            bin_point_indices = torch.nonzero(bin_mask, as_tuple=True)[0]
            bin_distances = distances[bin_mask]
            
            # Calculate average distance for this bin
            avg_distance = bin_distances.mean().item()
            
            # Get sampling rate based on average distance
            sampling_rate = self._calculate_sampling_rate(avg_distance)
            
            # Calculate how many points to keep from this bin
            bin_size = len(bin_point_indices)
            points_to_keep = max(1, int(bin_size * sampling_rate))
            
            # Sample from bin
            if points_to_keep >= bin_size:
                # Keep all points in bin
                selected_indices.extend(bin_point_indices.tolist())
            else:
                # Randomly sample points (uniform within bin)
                perm = torch.randperm(bin_size, device=device)[:points_to_keep]
                selected_indices.extend(bin_point_indices[perm].tolist())
                
        return torch.tensor(selected_indices, device=device, dtype=torch.long)
