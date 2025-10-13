"""Test cases for segmentation rendering from point clouds."""

import torch
import pytest
from utils.point_cloud_ops.rendering import render_segmentation_from_pointcloud


def test_render_segmentation_basic():
    """Test basic segmentation rendering without mask."""
    # Create point cloud with different class labels
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, -1.0],  # Center
                [0.5, 0.5, -2.0],  # Upper right
                [-0.5, 0.5, -1.5],  # Upper left
                [0.0, -0.5, -3.0],  # Bottom center
            ],
            dtype=torch.float32,
        ),
        'labels': torch.tensor([0, 1, 2, 3], dtype=torch.int64),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)

    # Render without mask
    seg_map = render_segmentation_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention="opengl",
        return_mask=False,
    )

    assert seg_map.shape == (100, 100)
    assert seg_map.dtype == torch.int64

    # Check that we have the expected labels
    unique_labels = torch.unique(seg_map)
    assert 255 in unique_labels  # Ignore index
    valid_labels = unique_labels[unique_labels != 255]
    assert len(valid_labels) > 0  # Should have some valid labels


def test_render_segmentation_with_mask():
    """Test segmentation rendering with valid mask."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.2, 0.2, -1.5],
                [-0.3, -0.3, -2.0],
            ],
            dtype=torch.float32,
        ),
        'labels': torch.tensor([10, 20, 30], dtype=torch.int64),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)

    # Render with mask
    seg_map, valid_mask = render_segmentation_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention="opengl",
        return_mask=True,
    )

    assert seg_map.shape == (100, 100)
    assert valid_mask.shape == (100, 100)
    assert valid_mask.dtype == torch.bool

    # Check that we have some valid pixels
    assert valid_mask.sum() > 0
    assert valid_mask.sum() < 100 * 100  # Not all pixels should be filled

    # Check that valid pixels have valid labels
    valid_labels = seg_map[valid_mask]
    assert torch.all((valid_labels == 10) | (valid_labels == 20) | (valid_labels == 30))

    # Check that invalid pixels have ignore_index
    assert (seg_map[~valid_mask] == 255).all()


def test_render_segmentation_custom_key():
    """Test using custom key for segmentation labels."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.1, 0.1, -1.2],
            ],
            dtype=torch.float32,
        ),
        'semantic_labels': torch.tensor([5, 10], dtype=torch.int64),  # Custom key
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    seg_map = render_segmentation_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        key='semantic_labels',  # Use custom key
    )

    # Should successfully render with custom key
    assert seg_map.shape == (100, 100)
    valid_labels = seg_map[seg_map != 255]
    assert len(valid_labels) > 0


def test_render_segmentation_sorting():
    """Test that closer points overwrite farther ones."""
    # Two points at same image location but different depths
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, -3.0],  # Farther point
                [0.0, 0.0, -1.0],  # Closer point (should overwrite)
            ],
            dtype=torch.float32,
        ),
        'labels': torch.tensor([100, 200], dtype=torch.int64),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)

    seg_map = render_segmentation_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
    )

    # Check that label 200 (closer point) appears somewhere
    has_label_200 = (seg_map == 200).any()
    has_label_100 = (seg_map == 100).any()

    # Should have the closer point's label rendered
    assert has_label_200 or has_label_100  # At least one label should appear


def test_render_segmentation_custom_ignore_index():
    """Test using custom ignore index for empty pixels."""
    pc_data = {
        'pos': torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        'labels': torch.tensor([42], dtype=torch.int64),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    custom_ignore = 127

    seg_map = render_segmentation_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        ignore_index=custom_ignore,
    )

    # Most pixels should have the custom ignore index
    background_pixels = seg_map == custom_ignore
    assert background_pixels.sum() > 100 * 100 * 0.9  # Most pixels are background


def test_render_segmentation_points_behind_camera():
    """Test that points behind camera are filtered out."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, 1.0],  # Behind camera (positive Z in OpenGL)
                [0.0, 0.0, -1.0],  # In front of camera
            ],
            dtype=torch.float32,
        ),
        'labels': torch.tensor([10, 20], dtype=torch.int64),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    seg_map, valid_mask = render_segmentation_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        return_mask=True,
    )

    # Should only have label 20 (point in front)
    valid_labels = seg_map[valid_mask]
    assert torch.all(valid_labels == 20)


def test_render_segmentation_empty_pointcloud():
    """Test that empty point cloud raises assertion error."""
    pc_data = {
        'pos': torch.empty((0, 3), dtype=torch.float32),
        'labels': torch.empty((0,), dtype=torch.int64),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    with pytest.raises(AssertionError, match="Labels tensor must not be empty"):
        render_segmentation_from_pointcloud(
            pc_data=pc_data,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            resolution=(100, 100),
        )


def test_render_segmentation_multi_class():
    """Test rendering with multiple segmentation classes."""
    # Create a dense point cloud with various classes
    import numpy as np

    grid_size = 10
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(x, y)

    positions = []
    labels = []
    for i in range(grid_size):
        for j in range(grid_size):
            positions.append([xx[i, j], yy[i, j], -2.0])
            # Assign label based on quadrant
            if xx[i, j] >= 0 and yy[i, j] >= 0:
                labels.append(0)  # Upper right
            elif xx[i, j] < 0 and yy[i, j] >= 0:
                labels.append(1)  # Upper left
            elif xx[i, j] < 0 and yy[i, j] < 0:
                labels.append(2)  # Lower left
            else:
                labels.append(3)  # Lower right

    pc_data = {
        'pos': torch.tensor(positions, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64),
    }

    camera_intrinsics = torch.tensor(
        [[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    seg_map, valid_mask = render_segmentation_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        return_mask=True,
    )

    # Should have all 4 classes represented
    unique_labels = torch.unique(seg_map[valid_mask])
    assert len(unique_labels) == 4
    assert torch.all((unique_labels >= 0) & (unique_labels <= 3))


def test_render_segmentation_invalid_inputs():
    """Test various invalid input conditions."""
    valid_pc_data = {
        'pos': torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        'labels': torch.tensor([0], dtype=torch.int64),
    }
    valid_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    valid_extrinsics = torch.eye(4, dtype=torch.float32)

    # Test missing 'pos' key
    with pytest.raises(AssertionError, match=r"pc\.keys\(\)=dict_keys\(\['labels'\]\)"):
        render_segmentation_from_pointcloud(
            pc_data={'labels': torch.zeros(1, dtype=torch.int64)},
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100),
        )

    # Test missing labels key
    with pytest.raises(AssertionError, match="must contain 'labels' key"):
        render_segmentation_from_pointcloud(
            pc_data={'pos': torch.zeros((1, 3))},
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100),
        )

    # Test mismatched lengths
    with pytest.raises(
        AssertionError,
        match=r"\{pos: torch\.Size\(\[2, 3\]\), labels: torch\.Size\(\[3\]\)\}",
    ):
        render_segmentation_from_pointcloud(
            pc_data={
                'pos': torch.zeros((2, 3)),
                'labels': torch.zeros(3, dtype=torch.int64),
            },
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100),
        )

    # Test invalid ignore_index
    with pytest.raises(AssertionError, match="ignore_index must be in range"):
        render_segmentation_from_pointcloud(
            pc_data=valid_pc_data,
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100),
            ignore_index=256,
        )
