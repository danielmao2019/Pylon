"""Continuous Level of Detail system for point cloud visualization."""
from typing import Any, Dict, Tuple
import torch

from data.transforms.vision_3d.pclod.lod_utils import get_camera_position
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.select import Select
import logging

logger = logging.getLogger(__name__)


class ContinuousLOD:
    """Continuous Level of Detail using camera distance-based point sampling.

    Supports two modes:
    1. Spatial binning: Groups points by camera-relative position for uniform coverage
    2. Simple distance: Direct distance-based sampling for maximum performance
    """

    def __init__(
        self,
        spatial_bins: int = 64,
        near_distance_factor: float = 0.75,
        far_distance_factor: float = 6.0,
        near_sampling_rate: float = 0.9,
        far_sampling_rate: float = 0.1,
        use_spatial_binning: bool = True
    ):
        """Initialize continuous LOD system.

        Args:
            spatial_bins: Number of spatial bins for coverage (default: 64)
            near_distance_factor: Factor of diagonal size for "near" distance (default: 0.75)
            far_distance_factor: Factor of diagonal size for "far" distance (default: 6.0)
            near_sampling_rate: Sampling rate for near bins (default: 0.9 = keep 90%)
            far_sampling_rate: Sampling rate for far bins (default: 0.1 = keep 10%)
            use_spatial_binning: Whether to use spatial binning (default: True)
        """
        self.spatial_bins = spatial_bins
        self.near_distance_factor = near_distance_factor
        self.far_distance_factor = far_distance_factor
        self.near_sampling_rate = near_sampling_rate
        self.far_sampling_rate = far_sampling_rate
        self.use_spatial_binning = use_spatial_binning

    # =============================================================================
    # Public API
    # =============================================================================

    def subsample(
        self,
        point_cloud: PointCloud,
        camera_state: Dict[str, Any]
    ) -> PointCloud:
        """Subsample point cloud using continuous LOD approach.

        Args:
            point_cloud: Point cloud with xyz coordinates and optional features
            camera_state: Camera state with 'eye', 'center', 'up'

        Returns:
            Subsampled point cloud dictionary
        """
        assert isinstance(point_cloud, PointCloud), f"{type(point_cloud)=}"

        points = point_cloud.xyz
        camera_pos = get_camera_position(camera_state, device=points.device, dtype=points.dtype)
        distances = torch.norm(points - camera_pos, dim=1)

        # Calculate relative distance thresholds based on point cloud size
        diagonal_size = self._calculate_diagonal_size(points)
        near_distance = diagonal_size * self.near_distance_factor
        far_distance = diagonal_size * self.far_distance_factor

        if self.use_spatial_binning:
            selected_indices = self._spatial_binning_pipeline(points, camera_state, distances, near_distance, far_distance)
        else:
            selected_indices = self._simple_distance_pipeline(distances, near_distance, far_distance)

        # Log LOD information
        original_count = point_cloud.num_points
        subsampled_count = len(selected_indices)
        logger.info(f"Continuous LOD: Points={subsampled_count}/{original_count} ({100*subsampled_count/original_count:.1f}%), "
                   f"diagonal={diagonal_size:.2f}, near={near_distance:.2f}, far={far_distance:.2f}")

        return Select(selected_indices)(point_cloud)

    # =============================================================================
    # Main Processing Pipelines
    # =============================================================================

    def _spatial_binning_pipeline(
        self,
        points: torch.Tensor,
        camera_state: Dict[str, Any],
        distances: torch.Tensor,
        near_distance: float,
        far_distance: float
    ) -> torch.Tensor:
        """Complete spatial binning pipeline with distance-based sampling."""
        # Step 1: Transform to camera space and create spatial bins
        coords_cam = self._transform_to_camera_space(points, camera_state)
        bin_indices = self._coords_to_bin_indices(coords_cam)

        # Step 2: Apply per-bin distance-based sampling
        return self._vectorized_bin_sampling(distances, bin_indices, near_distance, far_distance)

    def _simple_distance_pipeline(self, distances: torch.Tensor, near_distance: float, far_distance: float) -> torch.Tensor:
        """Simple distance-based sampling without spatial binning for maximum performance."""
        return self._vectorized_distance_sampling(distances, near_distance, far_distance)

    # =============================================================================
    # Core Sampling Operations
    # =============================================================================

    def _vectorized_distance_sampling(self, distances: torch.Tensor, near_distance: float, far_distance: float) -> torch.Tensor:
        """Apply vectorized distance-based sampling to all points."""
        sampling_rates = self._calculate_sampling_rates(distances, near_distance, far_distance)
        random_values = torch.rand(len(distances), device=distances.device)
        selected_mask = random_values < sampling_rates
        return torch.nonzero(selected_mask, as_tuple=True)[0]

    def _vectorized_bin_sampling(self, distances: torch.Tensor, bin_indices: torch.Tensor, near_distance: float, far_distance: float) -> torch.Tensor:
        """Apply truly vectorized per-bin distance-based sampling."""
        device = distances.device

        # Calculate per-bin average distances using scatter operations
        unique_bins, inverse_indices = torch.unique(bin_indices, return_inverse=True)
        num_bins = len(unique_bins)

        # Compute average distance per bin using scatter_add
        bin_distance_sums = torch.zeros(num_bins, device=device)
        bin_counts = torch.zeros(num_bins, device=device)

        bin_distance_sums.scatter_add_(0, inverse_indices, distances)
        bin_counts.scatter_add_(0, inverse_indices, torch.ones_like(distances))

        # Calculate average distances and sampling rates for all bins
        bin_avg_distances = bin_distance_sums / torch.clamp(bin_counts, min=1)

        # Reuse existing sampling rate calculation method
        bin_sampling_rates = self._calculate_sampling_rates(bin_avg_distances, near_distance, far_distance)

        # Map sampling rates back to points
        point_sampling_rates = bin_sampling_rates[inverse_indices]

        # Apply vectorized random sampling
        random_values = torch.rand(len(distances), device=device)
        selected_mask = random_values < point_sampling_rates

        return torch.nonzero(selected_mask, as_tuple=True)[0]

    def _calculate_sampling_rates(self, distances: torch.Tensor, near_distance: float, far_distance: float) -> torch.Tensor:
        """Calculate sampling rates for distances using linear interpolation."""
        clamped_distances = torch.clamp(distances, near_distance, far_distance)
        t = (clamped_distances - near_distance) / (far_distance - near_distance)
        return self.near_sampling_rate + t * (self.far_sampling_rate - self.near_sampling_rate)

    # =============================================================================
    # Utility Methods
    # =============================================================================

    def _calculate_diagonal_size(self, points: torch.Tensor) -> float:
        """Calculate the diagonal size of the point cloud bounding box.

        Args:
            points: Point cloud positions (N, 3)

        Returns:
            Diagonal size of the bounding box
        """
        if len(points) == 0:
            return 1.0  # Default size for empty point clouds

        pc_min = points.min(dim=0)[0]
        pc_max = points.max(dim=0)[0]
        extents = pc_max - pc_min
        diagonal_size = torch.norm(extents).item()

        # Ensure minimum size to avoid division by zero
        return max(diagonal_size, 1e-6)

    # =============================================================================
    # Spatial Binning Utilities
    # =============================================================================

    def _transform_to_camera_space(
        self,
        points: torch.Tensor,
        camera_state: Dict[str, Any]
    ) -> torch.Tensor:
        """Transform points to camera-aligned coordinate system."""
        device, dtype = points.device, points.dtype
        camera_pos = get_camera_position(camera_state, device=device, dtype=dtype)
        right, up_corrected, forward = self._get_camera_basis(camera_state, device, dtype, camera_pos)

        # Project points onto camera basis vectors
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
        """Compute orthonormal camera coordinate system basis vectors."""
        center = camera_state.get('center', {'x': 0, 'y': 0, 'z': 0})
        up = camera_state.get('up', {'x': 0, 'y': 0, 'z': 1})

        center_pos = torch.tensor([center['x'], center['y'], center['z']], device=device, dtype=dtype)
        up_vec = torch.tensor([up['x'], up['y'], up['z']], device=device, dtype=dtype)

        # Create orthonormal basis using Gram-Schmidt process
        forward = center_pos - camera_pos
        forward = forward / torch.norm(forward)
        right = torch.cross(forward, up_vec)
        right = right / torch.norm(right)
        up_corrected = torch.cross(right, forward)

        return right, up_corrected, forward

    def _coords_to_bin_indices(self, coords_cam: torch.Tensor) -> torch.Tensor:
        """Convert camera-space coordinates to 1D bin indices using efficient GPU operations."""
        bins_per_dim = max(1, int(round(self.spatial_bins ** (1/3))))

        # Use percentile-based binning for better GPU performance and outlier handling
        percentiles = torch.quantile(coords_cam, torch.tensor([0.05, 0.95], device=coords_cam.device), dim=0)
        min_coords, max_coords = percentiles[0], percentiles[1]
        coord_range = torch.clamp(max_coords - min_coords, min=1e-6)

        # Map coordinates to bin indices
        normalized_coords = torch.clamp((coords_cam - min_coords) / coord_range, 0, 1)
        bin_coords = torch.clamp(
            (normalized_coords * bins_per_dim).long(),
            0, bins_per_dim - 1
        )

        # Convert 3D bin coordinates to 1D index
        return (
            bin_coords[:, 0] * bins_per_dim * bins_per_dim +
            bin_coords[:, 1] * bins_per_dim +
            bin_coords[:, 2]
        )
