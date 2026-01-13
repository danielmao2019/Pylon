"""Generate synthetic point cloud registration data for benchmarking."""

import numpy as np
import torch
from typing import Dict, List, Any
from utils.three_d.rotation.rodrigues import rodrigues_to_matrix
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def generate_point_cloud(num_points: int = 5000) -> torch.Tensor:
    """Generate a synthetic point cloud.

    Args:
        num_points: Number of points to generate

    Returns:
        Point cloud tensor of shape (num_points, 3)
    """
    # Generate points in a unit sphere with some structure
    # Core sphere
    core_points = torch.randn(num_points // 2, 3) * 0.3

    # Outer shell
    shell_points = torch.randn(num_points // 2, 3)
    shell_points = torch.nn.functional.normalize(shell_points, p=2, dim=1) * (0.5 + torch.rand(num_points // 2, 1) * 0.3)

    points = torch.cat([core_points, shell_points], dim=0)
    return points.float()


def generate_correspondences(num_points: int) -> torch.Tensor:
    """Generate synthetic correspondences between point clouds.

    Args:
        num_points: Number of points in each cloud

    Returns:
        Correspondence indices of shape (num_correspondences, 2)
    """
    num_correspondences = min(1000, num_points // 5)
    correspondences = torch.randint(0, num_points, (num_correspondences, 2))
    return correspondences


def generate_rigid_transform() -> Dict[str, torch.Tensor]:
    """Generate a random rigid transformation (rotation + translation).

    Returns:
        Dictionary with 'rotation' (3x3) and 'translation' (3,) tensors
    """
    # Random rotation using Rodrigues representation
    axis = torch.randn(3)
    axis = axis / torch.norm(axis)  # Normalize to unit vector
    angle = torch.rand(1) * np.pi / 4  # Up to 45 degrees
    angle = angle.squeeze()

    # Convert to rotation matrix using rodrigues_to_matrix utility
    rotation = rodrigues_to_matrix(axis, angle)

    # Random translation
    translation = torch.randn(3) * 0.5

    return {
        'rotation': rotation,
        'translation': translation
    }


def generate_mock_transforms() -> List[Dict[str, Any]]:
    """Generate mock transform options for the viewer.

    Returns:
        List of transform dictionaries
    """
    transforms = [
        {'index': 0, 'name': 'Original', 'type': 'identity'},
        {'index': 1, 'name': 'Noise 0.01', 'type': 'noise', 'std': 0.01},
        {'index': 2, 'name': 'Noise 0.05', 'type': 'noise', 'std': 0.05},
        {'index': 3, 'name': 'Downsample 0.8', 'type': 'downsample', 'ratio': 0.8},
        {'index': 4, 'name': 'Downsample 0.5', 'type': 'downsample', 'ratio': 0.5},
    ]
    return transforms


def generate_synthetic_pcr_dataset(
    num_datapoints: int = 100,
    num_points: int = 5000
) -> Dict[str, Any]:
    """Generate a complete synthetic PCR dataset for benchmarking.

    Args:
        num_datapoints: Number of datapoint pairs in the dataset
        num_points: Number of points per cloud

    Returns:
        Dictionary containing the complete dataset
    """
    dataset = {
        'datapoints': [],
        'transforms': generate_mock_transforms(),
        'class_labels': None,  # PCR doesn't use class labels
        'dataset_info': {
            'name': 'Synthetic_PCR_Benchmark',
            'type': 'pcr',
            'num_datapoints': num_datapoints,
            'num_points_per_cloud': num_points
        }
    }

    # Generate each datapoint
    for i in range(num_datapoints):
        # Generate source point cloud
        source_pc = generate_point_cloud(num_points)

        # Generate target point cloud with known transformation
        transform = generate_rigid_transform()
        target_pc = torch.matmul(source_pc, transform['rotation'].T) + transform['translation']

        # Add noise to target
        target_pc += torch.randn_like(target_pc) * 0.02

        # Generate correspondences
        correspondences = generate_correspondences(num_points)

        datapoint = {
            'index': i,
            'source_pc': source_pc,
            'target_pc': target_pc,
            'correspondences': correspondences,
            'ground_truth_transform': transform
        }

        dataset['datapoints'].append(datapoint)

    return dataset


def apply_synthetic_transform(point_cloud: torch.Tensor, transform_type: str, **kwargs) -> torch.Tensor:
    """Apply a synthetic transform to a point cloud.

    Args:
        point_cloud: Input point cloud of shape (N, 3)
        transform_type: Type of transform to apply
        **kwargs: Transform-specific parameters

    Returns:
        Transformed point cloud
    """
    if transform_type == 'identity':
        return point_cloud.clone()

    elif transform_type == 'noise':
        std = kwargs.get('std', 0.01)
        noise = torch.randn_like(point_cloud) * std
        return point_cloud + noise

    elif transform_type == 'downsample':
        ratio = kwargs.get('ratio', 0.8)
        num_keep = int(len(point_cloud) * ratio)
        indices = torch.randperm(len(point_cloud))[:num_keep]
        return point_cloud[indices]

    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def get_mock_datapoint(
    dataset: Dict[str, Any],
    index: int,
    transform_indices: List[int]
) -> Dict[str, Any]:
    """Get a datapoint from the synthetic dataset with specified transforms applied.

    Args:
        dataset: Synthetic dataset dictionary
        index: Datapoint index
        transform_indices: List of transform indices to apply

    Returns:
        Processed datapoint dictionary
    """
    if index >= len(dataset['datapoints']):
        index = index % len(dataset['datapoints'])

    base_datapoint = dataset['datapoints'][index]
    transforms = dataset['transforms']

    # Apply transforms to source and target point clouds
    source_pc = base_datapoint['source_pc']
    target_pc = base_datapoint['target_pc']

    for transform_idx in transform_indices:
        if transform_idx < len(transforms):
            transform_info = transforms[transform_idx]
            transform_type = transform_info['type']

            if transform_type != 'identity':
                # Apply transform to both clouds
                source_pc = apply_synthetic_transform(source_pc, transform_type, **transform_info)
                target_pc = apply_synthetic_transform(target_pc, transform_type, **transform_info)

    source_pc_obj = PointCloud(xyz=source_pc)
    target_pc_obj = PointCloud(xyz=target_pc)

    return {
        'inputs': {
            'source_pc': source_pc_obj,
            'target_pc': target_pc_obj
        },
        'labels': {
            'correspondences': base_datapoint['correspondences'],
            'transform': base_datapoint['ground_truth_transform']
        },
        'meta_info': {
            'index': index,
            'applied_transforms': transform_indices,
            'source_num_points': source_pc_obj.num_points,
            'target_num_points': target_pc_obj.num_points
        }
    }
