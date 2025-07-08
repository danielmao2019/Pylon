"""Continuous Level of Detail system for point cloud visualization."""
from typing import Dict, Any, Tuple
import torch
from utils.input_checks.point_cloud import check_point_cloud
from utils.point_cloud_ops.select import Select
from data.viewer.utils.lod_utils import get_camera_position


class ContinuousLOD:
    """Continuous Level of Detail using camera frustum binning and distance weighting."""
    
    def __init__(
        self,
        spatial_bins: int = 64,
        near_distance: float = 5.0,
        far_distance: float = 50.0,
        near_sampling_rate: float = 0.9,
        far_sampling_rate: float = 0.1,
        use_spatial_binning: bool = True
    ):
        """Initialize continuous LOD system.
        
        Args:
            spatial_bins: Number of spatial bins for coverage (default: 64)
            near_distance: Distance considered "near" to camera (default: 5.0)
            far_distance: Distance considered "far" from camera (default: 50.0)
            near_sampling_rate: Sampling rate for near bins (default: 0.9 = keep 90%)
            far_sampling_rate: Sampling rate for far bins (default: 0.1 = keep 10%)
            use_spatial_binning: Whether to use spatial binning (default: True)
        """
        self.spatial_bins = spatial_bins
        self.near_distance = near_distance
        self.far_distance = far_distance
        self.near_sampling_rate = near_sampling_rate
        self.far_sampling_rate = far_sampling_rate
        self.use_spatial_binning = use_spatial_binning
        
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
        
        # Get camera position on same device as points
        camera_pos = get_camera_position(camera_state, device=points.device, dtype=points.dtype)
        
        # Calculate distances for all points
        distances = torch.norm(points - camera_pos, dim=1)
        
        if self.use_spatial_binning:
            # Perform spatial binning
            bin_indices = self._spatial_binning(points, camera_state)
            
            # Sample from bins based on distance
            selected_indices = self._distance_based_bin_sampling(
                distances, bin_indices
            )
        else:
            # Use simple distance-based sampling without binning for maximum performance
            selected_indices = self._simple_distance_sampling(distances)
        
        # Return subsampled point cloud using established point cloud operations
        return Select(selected_indices)(point_cloud)

    # Main processing methods (in order of execution)
    
    def _spatial_binning(
        self, 
        points: torch.Tensor, 
        camera_state: Dict[str, Any]
    ) -> torch.Tensor:
        """Assign points to spatial bins based on camera-relative coordinates."""
        # Transform points to camera-aligned coordinate system
        coords_cam = self._transform_to_camera_space(points, camera_state)
        
        # Convert to bin indices
        return self._coords_to_bin_indices(coords_cam)
        
    def _distance_based_bin_sampling(
        self,
        distances: torch.Tensor,
        bin_indices: torch.Tensor
    ) -> torch.Tensor:
        """Sample points from bins based on distance-based sampling rates."""
        device = distances.device
        
        # Calculate sampling rates for all points based on distance
        sampling_rates = self._calculate_sampling_rates_vectorized(distances)
        
        # Use vectorized random sampling instead of per-bin loops
        random_values = torch.rand(len(distances), device=device)
        selected_mask = random_values < sampling_rates
        
        return torch.nonzero(selected_mask, as_tuple=True)[0]
    
    def _simple_distance_sampling(self, distances: torch.Tensor) -> torch.Tensor:
        """Simple distance-based sampling without spatial binning for maximum performance."""
        # Calculate sampling rates for all points based on distance
        sampling_rates = self._calculate_sampling_rates_vectorized(distances)
        
        # Use vectorized random sampling
        random_values = torch.rand(len(distances), device=distances.device)
        selected_mask = random_values < sampling_rates
        
        return torch.nonzero(selected_mask, as_tuple=True)[0]
        
    # Helper methods for spatial binning
        
    def _transform_to_camera_space(
        self, 
        points: torch.Tensor, 
        camera_state: Dict[str, Any]
    ) -> torch.Tensor:
        """Transform points to camera-aligned coordinate system."""
        device = points.device
        dtype = points.dtype
        
        # Get camera coordinate system
        camera_pos = get_camera_position(camera_state, device=device, dtype=dtype)
        right, up_corrected, forward = self._get_camera_basis(camera_state, device, dtype, camera_pos)
        
        # Transform points to camera space
        points_cam = points - camera_pos.unsqueeze(0)
        x_cam = torch.sum(points_cam * right.unsqueeze(0), dim=1)
        y_cam = torch.sum(points_cam * up_corrected.unsqueeze(0), dim=1)
        z_cam = torch.sum(points_cam * forward.unsqueeze(0), dim=1)
        
        return torch.stack([x_cam, y_cam, z_cam], dim=1)
        
    def _get_camera_basis(
        self, 
        camera_state: Dict[str, Any], 
        device: torch.device, 
        dtype: torch.dtype,
        camera_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get camera coordinate system basis vectors."""
        center = camera_state.get('center', {'x': 0, 'y': 0, 'z': 0})
        up = camera_state.get('up', {'x': 0, 'y': 0, 'z': 1})
        
        center_pos = torch.tensor([center['x'], center['y'], center['z']], device=device, dtype=dtype)
        up_vec = torch.tensor([up['x'], up['y'], up['z']], device=device, dtype=dtype)
        
        # Create orthonormal basis
        forward = center_pos - camera_pos
        forward = forward / torch.norm(forward)
        right = torch.cross(forward, up_vec)
        right = right / torch.norm(right)
        up_corrected = torch.cross(right, forward)
        
        return right, up_corrected, forward
        
    def _coords_to_bin_indices(self, coords_cam: torch.Tensor) -> torch.Tensor:
        """Convert camera-space coordinates to bin indices using efficient GPU operations."""
        bins_per_dim = max(1, int(round(self.spatial_bins ** (1/3))))
        
        # Use percentile-based binning for better GPU performance
        # This avoids expensive min/max operations and handles outliers better
        percentiles = torch.quantile(coords_cam, torch.tensor([0.05, 0.95], device=coords_cam.device), dim=0)
        min_coords = percentiles[0]
        max_coords = percentiles[1]
        
        # Handle edge case where all points are in a plane
        coord_range = torch.clamp(max_coords - min_coords, min=1e-6)
        
        normalized_coords = torch.clamp((coords_cam - min_coords) / coord_range, 0, 1)
        bin_coords = torch.clamp(
            (normalized_coords * bins_per_dim).long(),
            0, bins_per_dim - 1
        )
        
        # Convert 3D bin coordinates to 1D bin indices using efficient tensor operations
        return (
            bin_coords[:, 0] * bins_per_dim * bins_per_dim +
            bin_coords[:, 1] * bins_per_dim +
            bin_coords[:, 2]
        )
        
            
    def _calculate_sampling_rates_vectorized(self, distances: torch.Tensor) -> torch.Tensor:
        """Calculate sampling rates for all distances using vectorized operations."""
        # Clamp distances to [near_distance, far_distance] range
        clamped_distances = torch.clamp(distances, self.near_distance, self.far_distance)
        
        # Linear interpolation using vectorized operations
        t = (clamped_distances - self.near_distance) / (self.far_distance - self.near_distance)
        sampling_rates = self.near_sampling_rate + t * (self.far_sampling_rate - self.near_sampling_rate)
        
        return sampling_rates
