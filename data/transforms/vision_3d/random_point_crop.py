from typing import Dict, Optional, Union, Sequence
import numpy as np
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud


class RandomPointCrop(BaseTransform):
    """Random crop point cloud from a viewpoint (GeoTransformer style).
    
    This transform replicates GeoTransformer's random_crop_point_cloud_with_point:
    - Samples random viewpoint in 3D space from 8 cardinal directions
    - Computes Euclidean distances from viewpoint to all points  
    - Keeps points closest to the viewpoint (simulates limited sensor range)
    - Creates more irregular cropping patterns than plane-based cropping
    """

    def __init__(self, keep_ratio: float = 0.7, viewpoint: Optional[Union[Sequence[Union[int, float]], np.ndarray, torch.Tensor]] = None, limit: float = 500.0):
        """Initialize RandomPointCrop transform.
        
        Args:
            keep_ratio: Fraction of points to keep after cropping (0.0 to 1.0)
            viewpoint: Optional fixed viewpoint (3,). If None, random viewpoint is generated.
            limit: Distance limit for random viewpoint generation
        """
        assert isinstance(keep_ratio, (int, float)), f"keep_ratio must be numeric, got {type(keep_ratio)}"
        assert 0.0 < keep_ratio <= 1.0, f"keep_ratio must be in (0, 1], got {keep_ratio}"
        assert isinstance(limit, (int, float)), f"limit must be numeric, got {type(limit)}"
        assert limit > 0, f"limit must be positive, got {limit}"
        
        self.keep_ratio = float(keep_ratio)
        self.limit = float(limit)
        
        if viewpoint is not None:
            # Normalize viewpoint to torch.Tensor of shape (3,)
            if isinstance(viewpoint, (list, tuple)):
                viewpoint = torch.tensor(viewpoint, dtype=torch.float32)
            elif isinstance(viewpoint, np.ndarray):
                viewpoint = torch.from_numpy(viewpoint).float()
            elif isinstance(viewpoint, torch.Tensor):
                viewpoint = viewpoint.float()
            else:
                raise TypeError(f"viewpoint must be Sequence, np.ndarray, or torch.Tensor, got {type(viewpoint)}")
            
            assert viewpoint.shape == (3,), f"viewpoint must have shape (3,), got {viewpoint.shape}"
            
        self.viewpoint = viewpoint

    def _call_single(self, pc: Dict[str, torch.Tensor], generator: torch.Generator) -> Dict[str, torch.Tensor]:
        """Apply random point-based cropping to point cloud.
        
        Args:
            pc: Point cloud dictionary with 'pos' key and optional feature keys
            generator: Random number generator for reproducible results
            
        Returns:
            Cropped point cloud dictionary
        """
        check_point_cloud(pc)
        
        positions = pc['pos']  # Shape: (N, 3)
        num_points = positions.shape[0]
        num_samples = int(torch.floor(torch.tensor(num_points * self.keep_ratio + 0.5)).item())
        
        # Assert generator and positions are on same device type
        assert positions.device.type == generator.device.type, f"positions device type {positions.device.type} != generator device type {generator.device.type}"
        
        # Generate or use provided viewpoint
        if self.viewpoint is None:
            viewpoint_tensor = self._random_sample_viewpoint(generator)
        else:
            # Align viewpoint to positions.device if needed
            if self.viewpoint.device != positions.device:
                self.viewpoint = self.viewpoint.to(positions.device)
            viewpoint_tensor = self.viewpoint
        
        # Compute Euclidean distances from viewpoint to all points
        # Following GeoTransformer: distances = np.linalg.norm(viewpoint - points, axis=1)
        distances = torch.norm(viewpoint_tensor.unsqueeze(0) - positions, dim=1)
        
        # Select points with smallest distances (closest to viewpoint)
        # Following GeoTransformer: sel_indices = np.argsort(distances)[:num_samples]
        _, sel_indices = torch.topk(distances, num_samples, largest=False)
        
        # Apply selection to all keys in point cloud
        cropped_pc = {}
        for key, tensor in pc.items():
            if key == 'pos':
                cropped_pc[key] = positions[sel_indices]
            else:
                # Assume features have same first dimension as positions
                cropped_pc[key] = tensor[sel_indices]
        
        return cropped_pc

    def _random_sample_viewpoint(self, generator: torch.Generator) -> torch.Tensor:
        """Sample random viewpoint from 8 cardinal directions.
        
        This replicates GeoTransformer's random_sample_viewpoint function exactly.
        
        Args:
            generator: Random number generator for reproducible results
            
        Returns:
            Random viewpoint coordinates, shape (3,) on same device as generator
        """
        # Following GeoTransformer logic exactly:
        # viewpoint = np.random.rand(3) + np.array([limit, limit, limit]) * np.random.choice([1.0, -1.0], size=3)
        
        # Generate random offset [0, 1] for each dimension on generator's device
        random_offset = torch.rand(3, generator=generator, device=generator.device)
        
        # Generate random signs for each dimension (8 cardinal directions)
        random_values = torch.rand(3, generator=generator, device=generator.device)
        signs = torch.where(random_values > 0.5, 1.0, -1.0)
        
        # Create viewpoint following GeoTransformer formula
        limit_tensor = torch.tensor([self.limit, self.limit, self.limit], device=generator.device, dtype=torch.float32)
        viewpoint = random_offset + limit_tensor * signs
        
        return viewpoint
