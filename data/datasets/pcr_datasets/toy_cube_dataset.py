"""Toy cube dataset for testing WebGL point cloud visualization."""

from typing import Tuple, Dict, Any
import numpy as np
import torch
from data.datasets.base_dataset import BaseDataset


class ToyCubeDataset(BaseDataset):
    """A toy PCR dataset containing just cube examples for testing visualization."""
    
    # Required class attributes from BaseDataset
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': 1, 'val': 0, 'test': 0}
    INPUT_NAMES = ['src_pc', 'tgt_pc']
    LABEL_NAMES = ['transform']
    SHA1SUM = None
    
    def __init__(self, cube_density: int = 8, **kwargs):
        """Initialize toy cube dataset.
        
        Args:
            cube_density: Number of points per cube edge
            **kwargs: Arguments passed to BaseDataset (including split)
        """
        self.cube_density = cube_density
        super().__init__(**kwargs)
        
    def _init_annotations(self) -> None:
        """Initialize annotations for the dataset."""
        if self.split == 'train':
            self.annotations = [{'cube_pair': 'source_target'}]
        else:
            self.annotations = []
    
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a single datapoint with source and target cubes."""
        
        # Create source cube (6 colored faces)
        src_points, src_colors = self._create_cube_points(center=[0, 0, 0])
        
        # Create target cube (same structure, different colors)
        tgt_points, tgt_colors = self._create_cube_points(
            center=[2, 1, 0.5],  # Translated
            color_shift=0.3      # Different colors
        )
        
        # Create ground truth transformation
        # Rotation: 30 degrees around Z axis + small translation
        angle = np.pi / 6  # 30 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        transform = torch.tensor([
            [cos_a, -sin_a, 0, 2.0],   # Rotation + X translation
            [sin_a,  cos_a, 0, 1.0],   # Rotation + Y translation
            [0,      0,     1, 0.5],   # Z translation
            [0,      0,     0, 1.0]    # Homogeneous
        ], dtype=torch.float32)
        
        # Create inputs dictionary (matching PCR dataset format)
        inputs = {
            'src_pc': {
                'pos': src_points,
                'feat': torch.ones((src_points.shape[0], 1), dtype=torch.float32),
                'rgb': src_colors
            },
            'tgt_pc': {
                'pos': tgt_points,
                'feat': torch.ones((tgt_points.shape[0], 1), dtype=torch.float32),
                'rgb': tgt_colors
            },
            'correspondences': torch.empty(0, 2, dtype=torch.long)  # Empty correspondences for toy dataset
        }
        
        # Create labels dictionary
        labels = {
            'transform': transform
        }
        
        # Create meta_info dictionary
        meta_info = {
            'src_points_count': len(src_points),
            'tgt_points_count': len(tgt_points),
            'transform_description': 'Rotation 30° + Translation [2, 1, 0.5]'
        }
        
        return inputs, labels, meta_info
    
    def _create_cube_points(self, center=[0, 0, 0], color_shift=0.0):
        """Create a cube with 6 different colored faces.
        
        Args:
            center: Center position of the cube
            color_shift: Shift colors by this amount (for variety)
            
        Returns:
            Tuple of (points, colors) as torch tensors
        """
        points = []
        colors = []
        
        # Define 6 face colors with optional shift
        base_colors = [
            [1, 0, 0],  # Red - Front face (z=1)
            [0, 1, 0],  # Green - Back face (z=0)  
            [0, 0, 1],  # Blue - Right face (x=1)
            [1, 1, 0],  # Yellow - Left face (x=0)
            [1, 0, 1],  # Magenta - Top face (y=1)
            [0, 1, 1],  # Cyan - Bottom face (y=0)
        ]
        
        # Apply color shift 
        face_colors = [[min(1.0, c + color_shift) for c in color] for color in base_colors]
        
        density = max(1, self.cube_density)  # Ensure minimum density of 1
        
        # Handle edge case of density=1
        if density == 1:
            step_values = [0.0]
        else:
            step_values = [i / (density - 1) for i in range(density)]
        
        # Generate points for each face
        for i in range(density):
            for j in range(density):
                u = step_values[i]  # 0 to 1
                v = step_values[j]  # 0 to 1
                
                # Front face (z=1) - Red
                points.append([center[0] + u, center[1] + v, center[2] + 1])
                colors.append(face_colors[0])
                
                # Back face (z=0) - Green
                points.append([center[0] + u, center[1] + v, center[2] + 0])
                colors.append(face_colors[1])
                
                # Right face (x=1) - Blue
                points.append([center[0] + 1, center[1] + u, center[2] + v])
                colors.append(face_colors[2])
                
                # Left face (x=0) - Yellow
                points.append([center[0] + 0, center[1] + u, center[2] + v])
                colors.append(face_colors[3])
                
                # Top face (y=1) - Magenta
                points.append([center[0] + u, center[1] + 1, center[2] + v])
                colors.append(face_colors[4])
                
                # Bottom face (y=0) - Cyan
                points.append([center[0] + u, center[1] + 0, center[2] + v])
                colors.append(face_colors[5])
        
        return torch.tensor(points, dtype=torch.float32), torch.tensor(colors, dtype=torch.float32)
