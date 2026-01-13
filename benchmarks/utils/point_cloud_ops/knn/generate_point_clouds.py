"""Point cloud generation utilities for KNN benchmarking."""

import torch
import numpy as np
from typing import List, Tuple


def generate_point_clouds(n_points: int, shapes: List[str], device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate different point cloud shapes for benchmarking.

    Args:
        n_points: Number of points in the point cloud
        shapes: List of shape types to generate
        device: Device to put tensors on

    Returns:
        List of (query_points, reference_points) tuples
    """
    point_clouds = []

    for shape in shapes:
        if shape == "uniform_cube":
            # Uniformly distributed points in a cube
            query = torch.rand(n_points, 3, device=device) * 100
            reference = torch.rand(n_points, 3, device=device) * 100

        elif shape == "gaussian_cluster":
            # Points clustered around origin with Gaussian distribution
            query = torch.randn(n_points, 3, device=device) * 10
            reference = torch.randn(n_points, 3, device=device) * 10

        elif shape == "sphere_surface":
            # Points on sphere surface
            # Generate random points on unit sphere using spherical coordinates
            theta = torch.rand(n_points, device=device) * 2 * np.pi
            phi = torch.acos(2 * torch.rand(n_points, device=device) - 1)
            r = 50  # Radius

            x = r * torch.sin(phi) * torch.cos(theta)
            y = r * torch.sin(phi) * torch.sin(theta)
            z = r * torch.cos(phi)
            query = torch.stack([x, y, z], dim=1)

            # Generate another set for reference
            theta = torch.rand(n_points, device=device) * 2 * np.pi
            phi = torch.acos(2 * torch.rand(n_points, device=device) - 1)
            x = r * torch.sin(phi) * torch.cos(theta)
            y = r * torch.sin(phi) * torch.sin(theta)
            z = r * torch.cos(phi)
            reference = torch.stack([x, y, z], dim=1)

        elif shape == "line_with_noise":
            # Points along a line with some noise
            t = torch.linspace(0, 100, n_points, device=device)
            noise = torch.randn(n_points, 3, device=device) * 2
            query = torch.stack([t, t * 0.5, t * 0.3], dim=1) + noise

            noise = torch.randn(n_points, 3, device=device) * 2
            reference = torch.stack([t, t * 0.5, t * 0.3], dim=1) + noise

        elif shape == "multiple_clusters":
            # Multiple separated clusters
            n_clusters = 5
            points_per_cluster = n_points // n_clusters
            remainder = n_points % n_clusters

            query_list = []
            reference_list = []

            for i in range(n_clusters):
                # Add remainder points to last cluster
                cluster_size = points_per_cluster + (remainder if i == n_clusters - 1 else 0)

                # Each cluster at different location
                center = torch.tensor([i * 30, i * 20, i * 10], device=device, dtype=torch.float32)
                query_cluster = torch.randn(cluster_size, 3, device=device) * 5 + center
                reference_cluster = torch.randn(cluster_size, 3, device=device) * 5 + center

                query_list.append(query_cluster)
                reference_list.append(reference_cluster)

            query = torch.cat(query_list, dim=0)
            reference = torch.cat(reference_list, dim=0)

        else:
            raise ValueError(f"Unknown shape: {shape}")

        # Ensure float32 dtype
        query = query.float()
        reference = reference.float()

        point_clouds.append((query, reference))

    return point_clouds
