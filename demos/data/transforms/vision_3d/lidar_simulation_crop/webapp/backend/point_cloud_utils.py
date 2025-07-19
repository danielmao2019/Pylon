#!/usr/bin/env python3
"""
Point cloud generation utilities for LiDAR simulation demos.
"""

import numpy as np
import torch
from typing import Literal


def create_toy_point_cloud(
    cloud_type: Literal['cube', 'sphere', 'scene'] = 'cube',
    num_points: int = 2000,
    noise: float = 0.02,
    seed: int = 42
) -> torch.Tensor:
    """Create synthetic point clouds for demonstration.
    
    Args:
        cloud_type: Type of point cloud to generate ('cube', 'sphere', or 'scene')
        num_points: Number of points to generate
        noise: Amount of noise to add to the points
        seed: Random seed for reproducible generation
        
    Returns:
        Point cloud as torch.Tensor [N, 3]
    """
    # Set seed for reproducible point cloud generation
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if cloud_type == 'cube':
        # Create a cube with points on surface and interior
        edge_points = []
        # Generate points on each face
        for face in range(6):
            n_face = num_points // 6
            if face == 0:  # +X face
                x = torch.ones(n_face) * 2.0
                y = torch.rand(n_face) * 4.0 - 2.0
                z = torch.rand(n_face) * 4.0 - 2.0
            elif face == 1:  # -X face
                x = torch.ones(n_face) * -2.0
                y = torch.rand(n_face) * 4.0 - 2.0
                z = torch.rand(n_face) * 4.0 - 2.0
            elif face == 2:  # +Y face
                x = torch.rand(n_face) * 4.0 - 2.0
                y = torch.ones(n_face) * 2.0
                z = torch.rand(n_face) * 4.0 - 2.0
            elif face == 3:  # -Y face
                x = torch.rand(n_face) * 4.0 - 2.0
                y = torch.ones(n_face) * -2.0
                z = torch.rand(n_face) * 4.0 - 2.0
            elif face == 4:  # +Z face
                x = torch.rand(n_face) * 4.0 - 2.0
                y = torch.rand(n_face) * 4.0 - 2.0
                z = torch.ones(n_face) * 2.0
            else:  # -Z face
                x = torch.rand(n_face) * 4.0 - 2.0
                y = torch.rand(n_face) * 4.0 - 2.0
                z = torch.ones(n_face) * -2.0
            
            face_points = torch.stack([x, y, z], dim=1)
            edge_points.append(face_points)
        
        points = torch.cat(edge_points, dim=0)
        
    elif cloud_type == 'sphere':
        # Create a sphere
        phi = torch.rand(num_points) * 2 * np.pi  # Azimuth
        theta = torch.rand(num_points) * np.pi    # Elevation
        r = torch.rand(num_points) ** (1/3) * 2.0  # Uniform volume distribution
        
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        
        points = torch.stack([x, y, z], dim=1)
        
    elif cloud_type == 'scene':
        # Create a simple scene with multiple objects
        # Ground plane
        ground_x = torch.rand(num_points // 4) * 20.0 - 10.0
        ground_y = torch.rand(num_points // 4) * 20.0 - 10.0
        ground_z = torch.zeros(num_points // 4) - 2.0
        
        # Building 1
        build1_x = torch.rand(num_points // 4) * 3.0 + 2.0
        build1_y = torch.rand(num_points // 4) * 3.0 - 1.5
        build1_z = torch.rand(num_points // 4) * 5.0 - 2.0
        
        # Building 2
        build2_x = torch.rand(num_points // 4) * 2.0 - 6.0
        build2_y = torch.rand(num_points // 4) * 4.0 + 1.0
        build2_z = torch.rand(num_points // 4) * 3.0 - 2.0
        
        # Tree-like structure
        tree_x = torch.rand(num_points // 4) * 1.0 - 0.5
        tree_y = torch.rand(num_points // 4) * 1.0 + 6.0
        tree_z = torch.rand(num_points // 4) * 4.0 - 2.0
        
        all_x = torch.cat([ground_x, build1_x, build2_x, tree_x])
        all_y = torch.cat([ground_y, build1_y, build2_y, tree_y])
        all_z = torch.cat([ground_z, build1_z, build2_z, tree_z])
        
        points = torch.stack([all_x, all_y, all_z], dim=1)
    
    else:
        raise ValueError(f"Unknown cloud type: {cloud_type}")
    
    # Add noise
    if noise > 0:
        points = points + torch.randn_like(points) * noise
    
    return points