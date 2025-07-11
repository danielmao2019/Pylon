from typing import Dict, Optional
import numpy as np
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud


class RandomPlaneCrop(BaseTransform):
    """Random crop point cloud with a plane (GeoTransformer style).
    
    This transform replicates GeoTransformer's random_crop_point_cloud_with_plane:
    - Generates random plane normal from unit sphere using spherical coordinates
    - Computes dot product distances from plane to all points
    - Keeps points with largest distances (one side of plane)
    - Preserves object topology better than point-based cropping
    """

    def __init__(self, keep_ratio: float = 0.7, plane_normal: Optional[np.ndarray] = None):
        """Initialize RandomPlaneCrop transform.
        
        Args:
            keep_ratio: Fraction of points to keep after cropping (0.0 to 1.0)
            plane_normal: Optional fixed plane normal (3,). If None, random normal is generated.
        """
        assert isinstance(keep_ratio, (int, float)), f"keep_ratio must be numeric, got {type(keep_ratio)}"
        assert 0.0 < keep_ratio <= 1.0, f"keep_ratio must be in (0, 1], got {keep_ratio}"
        
        self.keep_ratio = float(keep_ratio)
        self.plane_normal = plane_normal
        
        if plane_normal is not None:
            assert isinstance(plane_normal, np.ndarray), f"plane_normal must be np.ndarray, got {type(plane_normal)}"
            assert plane_normal.shape == (3,), f"plane_normal must have shape (3,), got {plane_normal.shape}"

    def _call_single(self, pc: Dict[str, torch.Tensor], generator: torch.Generator) -> Dict[str, torch.Tensor]:
        """Apply random plane cropping to point cloud.
        
        Args:
            pc: Point cloud dictionary with 'pos' key and optional feature keys
            generator: Random number generator for reproducible results
            
        Returns:
            Cropped point cloud dictionary
        """
        check_point_cloud(pc)
        
        positions = pc['pos']  # Shape: (N, 3)
        num_points = positions.shape[0]
        num_samples = int(np.floor(num_points * self.keep_ratio + 0.5))
        
        # Generate or use provided plane normal
        if self.plane_normal is None:
            plane_normal = self._random_sample_plane(generator)
        else:
            plane_normal = self.plane_normal.copy()
        
        # Convert to tensor on same device as positions
        plane_normal_tensor = torch.from_numpy(plane_normal).float().to(positions.device)
        
        # Compute distances from plane (dot product with normal)
        # Following GeoTransformer: distances = np.dot(points, p_normal)
        distances = torch.mm(positions, plane_normal_tensor.unsqueeze(1)).squeeze(1)
        
        # Select points with largest distances (one side of plane)
        # Following GeoTransformer: sel_indices = np.argsort(-distances)[:num_samples]
        _, sel_indices = torch.topk(distances, num_samples, largest=True)
        
        # Apply selection to all keys in point cloud
        cropped_pc = {}
        for key, tensor in pc.items():
            if key == 'pos':
                cropped_pc[key] = positions[sel_indices]
            else:
                # Assume features have same first dimension as positions
                cropped_pc[key] = tensor[sel_indices]
        
        return cropped_pc

    def _random_sample_plane(self, generator: torch.Generator) -> np.ndarray:
        """Generate random plane normal from unit sphere using spherical coordinates.
        
        This replicates GeoTransformer's random_sample_plane function exactly.
        
        Args:
            generator: Random number generator for reproducible results
            
        Returns:
            Random unit normal vector, shape (3,)
        """
        # Generate random spherical coordinates
        # Following GeoTransformer logic exactly
        phi = torch.rand(1, generator=generator).item() * 2 * np.pi  # longitude [0, 2π]
        theta = torch.rand(1, generator=generator).item() * np.pi     # latitude [0, π]
        
        # Convert spherical to Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi) 
        z = np.cos(theta)
        
        normal = np.array([x, y, z], dtype=np.float32)
        
        return normal