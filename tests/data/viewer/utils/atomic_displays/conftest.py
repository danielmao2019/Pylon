"""Shared fixtures for atomic displays tests.

CRITICAL: This conftest.py provides shared fixtures for all atomic display tests.
Uses pytest fixtures only - no class definitions as per CLAUDE.md guidelines.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any, List, Tuple


# ================================================================================
# Basic Tensor Fixtures
# ================================================================================

@pytest.fixture
def rgb_tensor():
    """RGB image tensor of shape [3, H, W]."""
    return torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)


@pytest.fixture
def grayscale_tensor():
    """Grayscale image tensor of shape [1, H, W]."""
    return torch.randint(0, 255, (1, 32, 32), dtype=torch.uint8)


@pytest.fixture
def depth_tensor():
    """Depth map tensor of shape [H, W] with realistic depth values."""
    return torch.rand(32, 32, dtype=torch.float32) * 10.0 + 0.1  # Range [0.1, 10.1]


@pytest.fixture
def normal_tensor():
    """Surface normal tensor of shape [3, H, W] with normalized vectors."""
    normals = torch.randn(3, 32, 32, dtype=torch.float32)
    # Normalize to unit vectors
    magnitude = torch.sqrt((normals ** 2).sum(dim=0, keepdim=True))
    magnitude = torch.clamp(magnitude, min=1e-8)  # Avoid division by zero
    return normals / magnitude


@pytest.fixture
def edge_tensor_2d():
    """Edge detection tensor of shape [H, W] with binary edge values."""
    return torch.randint(0, 2, (32, 32), dtype=torch.float32)


@pytest.fixture
def segmentation_tensor():
    """Semantic segmentation tensor of shape [H, W] with class indices."""
    return torch.randint(0, 5, (32, 32), dtype=torch.int64)


@pytest.fixture
def segmentation_tensor_3d():
    """Semantic segmentation tensor of shape [1, H, W] with class indices."""
    return torch.randint(0, 5, (1, 32, 32), dtype=torch.int64)


@pytest.fixture
def instance_surrogate_tensor():
    """Instance surrogate tensor of shape [2, H, W] with Y/X offsets."""
    # Create realistic offset values
    y_offsets = torch.randn(32, 32, dtype=torch.float32) * 5.0
    x_offsets = torch.randn(32, 32, dtype=torch.float32) * 5.0

    # Add some ignore regions (set to ignore_value=250)
    ignore_mask = torch.rand(32, 32) < 0.1  # 10% ignore regions
    y_offsets[ignore_mask] = 250
    x_offsets[ignore_mask] = 250

    return torch.stack([y_offsets, x_offsets], dim=0)


@pytest.fixture
def point_cloud_3d():
    """3D point cloud tensor of shape [N, 3]."""
    return torch.randn(1000, 3, dtype=torch.float32)


@pytest.fixture
def point_cloud_colors():
    """Point cloud colors tensor of shape [N, 3]."""
    return torch.randint(0, 255, (1000, 3), dtype=torch.uint8)


@pytest.fixture
def point_cloud_labels():
    """Point cloud labels tensor of shape [N]."""
    return torch.randint(0, 5, (1000,), dtype=torch.long)


# ================================================================================
# Camera and Display Fixtures
# ================================================================================

@pytest.fixture
def camera_state():
    """Camera state for LOD testing."""
    return {
        'eye': {'x': 1.0, 'y': 1.0, 'z': 1.0},
        'center': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'up': {'x': 0.0, 'y': 0.0, 'z': 1.0}
    }


@pytest.fixture
def class_labels():
    """Class labels mapping for segmentation visualization."""
    return {
        "segmentation": ["background", "building", "road", "vegetation", "water"]
    }


# ================================================================================
# Dictionary-based Fixtures for Complex Cases
# ================================================================================

@pytest.fixture
def segmentation_dict():
    """Dictionary-based segmentation format with masks and indices."""
    # Create 3 instance masks using bool (proper semantic type for masks)
    masks = []
    for i in range(3):
        mask = torch.zeros(32, 32, dtype=torch.bool)
        # Create a simple rectangular mask for each instance
        y_start, y_end = i * 8, (i + 1) * 8
        x_start, x_end = i * 8, (i + 1) * 8
        mask[y_start:y_end, x_start:x_end] = True
        masks.append(mask)

    indices = [0, 1, 2]  # Instance indices

    return {
        'masks': masks,
        'indices': indices
    }


# ================================================================================
# Edge Case Fixtures
# ================================================================================

@pytest.fixture
def empty_tensor():
    """Empty tensor for edge case testing."""
    return torch.empty((3, 0, 0), dtype=torch.float32)


@pytest.fixture
def single_pixel_tensor():
    """Single pixel tensor for edge case testing."""
    return torch.ones((3, 1, 1), dtype=torch.float32)


@pytest.fixture
def large_tensor():
    """Large tensor for performance testing."""
    return torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)


@pytest.fixture
def extreme_values_tensor():
    """Tensor with extreme values for robustness testing."""
    tensor = torch.zeros(3, 32, 32, dtype=torch.float32)
    tensor[0] = 1e6   # Very large values
    tensor[1] = 1e-6  # Very small values
    tensor[2] = -1e3  # Negative values
    return tensor


@pytest.fixture
def nan_tensor():
    """Tensor with NaN values for edge case testing."""
    tensor = torch.randn(3, 32, 32, dtype=torch.float32)
    tensor[0, :5, :5] = float('nan')  # Add some NaN values
    return tensor


@pytest.fixture
def inf_tensor():
    """Tensor with infinity values for edge case testing."""
    tensor = torch.randn(3, 32, 32, dtype=torch.float32)
    tensor[0, :5, :5] = float('inf')   # Add some positive infinity
    tensor[1, :5, :5] = float('-inf')  # Add some negative infinity
    return tensor


# ================================================================================
# Parametrized Fixtures
# ================================================================================

@pytest.fixture(params=["Viridis", "Plasma", "Inferno", "Cividis", "Gray"])
def colorscale(request):
    """Parametrized colorscale fixture for testing different visualization styles."""
    return request.param


@pytest.fixture(params=[torch.uint8, torch.float32, torch.int64])
def tensor_dtype(request):
    """Parametrized dtype fixture for testing different tensor types."""
    return request.param


@pytest.fixture(params=[(16, 16), (32, 32), (64, 64), (128, 128)])
def tensor_size(request):
    """Parametrized size fixture for testing different tensor dimensions."""
    return request.param


# ================================================================================
# LOD Testing Fixtures
# ================================================================================

@pytest.fixture
def lod_settings():
    """Level-of-Detail settings for point cloud testing."""
    return {
        'max_points': 5000,
        'min_points': 100,
        'distance_threshold': 10.0
    }


@pytest.fixture
def point_cloud_id():
    """Point cloud ID for discrete LOD testing."""
    return "test_point_cloud_123"


# ================================================================================
# Performance Testing Fixtures
# ================================================================================

@pytest.fixture(params=[100, 1000, 10000])
def point_cloud_sizes(request):
    """Parametrized point cloud sizes for performance testing."""
    n_points = request.param
    return torch.randn(n_points, 3, dtype=torch.float32)


@pytest.fixture(params=[(64, 64), (256, 256), (512, 512)])
def image_sizes(request):
    """Parametrized image sizes for performance testing."""
    h, w = request.param
    return torch.randint(0, 255, (3, h, w), dtype=torch.uint8)


# ================================================================================
# Batch Support Fixtures - CRITICAL for eval viewer
# ================================================================================

@pytest.fixture
def batched_rgb_tensor():
    """Batched RGB image tensor of shape [1, 3, H, W]."""
    return torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)


@pytest.fixture
def batched_grayscale_tensor():
    """Batched grayscale image tensor of shape [1, 1, H, W]."""
    return torch.randint(0, 255, (1, 1, 32, 32), dtype=torch.uint8)


@pytest.fixture
def batched_depth_tensor():
    """Batched depth map tensor of shape [1, H, W] with realistic depth values."""
    return torch.rand(1, 32, 32, dtype=torch.float32) * 10.0 + 0.1  # [1, H, W]


@pytest.fixture
def batched_normal_tensor():
    """Batched surface normal tensor of shape [1, 3, H, W] with normalized vectors."""
    normals = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    # Normalize to unit vectors
    magnitude = torch.sqrt((normals ** 2).sum(dim=1, keepdim=True))
    magnitude = torch.clamp(magnitude, min=1e-8)  # Avoid division by zero
    return normals / magnitude


@pytest.fixture
def batched_edge_tensor():
    """Batched edge detection tensor of shape [1, H, W] with binary edge values."""
    return torch.randint(0, 2, (1, 32, 32), dtype=torch.float32)


@pytest.fixture
def batched_segmentation_tensor():
    """Batched semantic segmentation tensor of shape [1, H, W] with class indices."""
    return torch.randint(0, 5, (1, 32, 32), dtype=torch.int64)


@pytest.fixture
def batched_instance_surrogate_tensor():
    """Batched instance surrogate tensor of shape [1, 2, H, W] with Y/X offsets."""
    # Create realistic offset values
    y_offsets = torch.randn(1, 32, 32, dtype=torch.float32) * 5.0
    x_offsets = torch.randn(1, 32, 32, dtype=torch.float32) * 5.0

    # Add some ignore regions (set to ignore_value=250)
    ignore_mask = torch.rand(32, 32) < 0.1  # 10% ignore regions
    y_offsets[0][ignore_mask] = 250
    x_offsets[0][ignore_mask] = 250

    return torch.stack([y_offsets[0], x_offsets[0]], dim=0).unsqueeze(0)  # [1, 2, H, W]
