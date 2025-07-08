"""Continuous Level of Detail system for point cloud visualization."""
from typing import Dict, Tuple, Any
import torch
from utils.input_checks.point_cloud import check_point_cloud


class ContinuousLOD:
    """Continuous Level of Detail using camera frustum binning and distance weighting."""
    
    def __init__(
        self,
        spatial_bins: int = 64,
        distance_exponent: float = 2.0,
        min_points: int = 2000,
        max_reduction: float = 0.8
    ):
        """Initialize continuous LOD system.
        
        Args:
            spatial_bins: Number of spatial bins for coverage (default: 64)
            distance_exponent: Distance weighting power (default: 2.0 for 1/d²)
            min_points: Minimum points to preserve (default: 2000)
            max_reduction: Maximum reduction ratio (default: 0.8 = 80% max reduction)
        """
        self.spatial_bins = spatial_bins
        self.distance_exponent = distance_exponent
        self.min_points = min_points
        self.max_reduction = max_reduction
        
    def calculate_target_points(
        self,
        point_cloud: Dict[str, torch.Tensor],
        camera_state: Dict[str, Any]
    ) -> int:
        """Calculate adaptive target points based on camera distance and point cloud size.
        
        Args:
            point_cloud: Dictionary with 'pos' key containing points
            camera_state: Camera state with 'eye', 'center', 'up'
            
        Returns:
            Target number of points for subsampling
        """
        points = point_cloud['pos']
        total_points = points.shape[0]
        
        # Get camera position and calculate average distance
        camera_pos = self._get_camera_position(camera_state).to(points.device)
        distances = torch.norm(points - camera_pos, dim=1)
        avg_distance = distances.mean().item()
        
        # Adaptive target based on distance and size
        if total_points < 10000:
            # Small clouds: minimal reduction
            target = int(total_points * 0.9)
        elif total_points < 50000:
            # Medium clouds: distance adaptive
            distance_factor = min(avg_distance / 10.0, 1.0)
            reduction = 0.3 + 0.4 * distance_factor  # 30-70% reduction
            target = int(total_points * (1.0 - reduction))
        else:
            # Large clouds: more aggressive
            distance_factor = min(avg_distance / 10.0, 1.0)
            reduction = 0.5 + 0.4 * distance_factor  # 50-90% reduction
            target = int(total_points * (1.0 - reduction))
        
        # Apply constraints
        target = self._apply_constraints(total_points, target)
        
        return target
        
    def subsample(
        self,
        point_cloud: Dict[str, torch.Tensor],
        camera_state: Dict[str, Any],
        target_points: int
    ) -> Dict[str, torch.Tensor]:
        """Subsample point cloud using continuous LOD approach.
        
        Args:
            point_cloud: Dictionary with 'pos' key and optional 'rgb', 'labels'
            camera_state: Camera state with 'eye', 'center', 'up'
            target_points: Target number of points after subsampling
            
        Returns:
            Subsampled point cloud dictionary
        """
        check_point_cloud(point_cloud)
        
        points = point_cloud['pos']
        current_points = points.shape[0]
        
        # Apply constraints
        target_points = self._apply_constraints(current_points, target_points)
        
        # If no reduction needed, return original
        if target_points >= current_points:
            return point_cloud
            
        # Get camera position
        camera_pos = self._get_camera_position(camera_state)
        
        # Calculate distance weights
        distance_weights = self._calculate_distance_weights(points, camera_pos)
        
        # Perform spatial binning
        bin_indices = self._spatial_binning(points, camera_state)
        
        # Sample from bins using distance weights
        selected_indices = self._weighted_bin_sampling(
            distance_weights, bin_indices, target_points
        )
        
        # Return subsampled point cloud
        return {key: tensor[selected_indices] for key, tensor in point_cloud.items()}
        
    def _apply_constraints(self, current_points: int, target_points: int) -> int:
        """Apply minimum points and maximum reduction constraints."""
        # Ensure minimum points
        target_points = max(target_points, self.min_points)
        
        # Ensure maximum reduction
        min_allowed = int(current_points * (1.0 - self.max_reduction))
        target_points = max(target_points, min_allowed)
        
        return target_points
        
    def _get_camera_position(self, camera_state: Dict[str, Any]) -> torch.Tensor:
        """Extract camera position from camera state."""
        eye = camera_state.get('eye', {'x': 1.5, 'y': 1.5, 'z': 1.5})
        return torch.tensor([eye['x'], eye['y'], eye['z']], dtype=torch.float32)
        
    def _calculate_distance_weights(
        self, 
        points: torch.Tensor, 
        camera_pos: torch.Tensor
    ) -> torch.Tensor:
        """Calculate distance-based importance weights (1/d^exponent)."""
        camera_pos = camera_pos.to(points.device)
        distances = torch.norm(points - camera_pos, dim=1)
        
        # Avoid division by zero
        epsilon = 1e-6
        weights = 1.0 / (distances ** self.distance_exponent + epsilon)
        
        return weights
        
    def _spatial_binning(
        self, 
        points: torch.Tensor, 
        camera_state: Dict[str, Any]
    ) -> torch.Tensor:
        """Assign points to spatial bins based on camera-relative coordinates."""
        device = points.device
        
        # Get camera vectors
        camera_pos = self._get_camera_position(camera_state).to(device)
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
        
    def _weighted_bin_sampling(
        self,
        distance_weights: torch.Tensor,
        bin_indices: torch.Tensor,
        target_points: int
    ) -> torch.Tensor:
        """Sample points from bins using distance weights."""
        device = distance_weights.device
        selected_indices = []
        
        unique_bins = torch.unique(bin_indices)
        total_weight = distance_weights.sum()
        
        for bin_id in unique_bins:
            bin_mask = (bin_indices == bin_id)
            if bin_mask.sum() == 0:
                continue
                
            # Get indices and weights for this bin
            bin_point_indices = torch.nonzero(bin_mask, as_tuple=True)[0]
            bin_weights = distance_weights[bin_mask]
            
            # Calculate points for this bin based on weight proportion
            bin_total_weight = bin_weights.sum()
            if total_weight > 0:
                points_for_bin = int(target_points * bin_total_weight / total_weight)
                points_for_bin = max(1, min(points_for_bin, len(bin_point_indices)))
            else:
                points_for_bin = 1
                
            # Sample from bin
            if points_for_bin >= len(bin_point_indices):
                # Take all points in bin
                selected_indices.extend(bin_point_indices.tolist())
            else:
                # Weighted sampling
                bin_probs = bin_weights / bin_weights.sum()
                sampled_indices = torch.multinomial(
                    bin_probs, points_for_bin, replacement=False
                )
                selected_indices.extend(bin_point_indices[sampled_indices].tolist())
                
        # Ensure we don't exceed target
        if len(selected_indices) > target_points:
            selected_indices = selected_indices[:target_points]
            
        # Fill up to target if needed
        if len(selected_indices) < target_points:
            all_indices = set(range(len(distance_weights)))
            available = list(all_indices - set(selected_indices))
            needed = min(target_points - len(selected_indices), len(available))
            if needed > 0:
                additional = torch.randperm(len(available), device=device)[:needed]
                selected_indices.extend([available[i] for i in additional])
                
        return torch.tensor(selected_indices, device=device, dtype=torch.long)