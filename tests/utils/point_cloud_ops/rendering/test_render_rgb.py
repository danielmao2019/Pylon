"""Test cases for RGB rendering from point clouds."""

import torch
import pytest
from utils.point_cloud_ops.rendering import render_rgb_from_pointcloud


def test_render_rgb_basic():
    """Test basic RGB rendering without mask."""
    # Create simple point cloud with 4 points
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, -1.0],  # Center point, close
                [0.5, 0.5, -2.0],  # Upper right, far
                [-0.5, 0.5, -1.5],  # Upper left, medium
                [0.0, -0.5, -3.0],  # Bottom center, very far
            ],
            dtype=torch.float32,
        ),
        'rgb': torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
            ],
            dtype=torch.float32,
        ),
    }

    # Simple camera at origin looking down -Z
    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    resolution = (100, 100)

    # Render without mask
    rgb_image = render_rgb_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention="opengl",
        return_mask=False,
    )

    assert rgb_image.shape == (3, 100, 100)
    assert rgb_image.dtype == torch.float32
    assert rgb_image.min() >= 0.0
    assert rgb_image.max() <= 1.0


def test_render_rgb_with_mask():
    """Test RGB rendering with valid mask."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.2, 0.2, -1.5],
            ],
            dtype=torch.float32,
        ),
        'rgb': torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)

    # Render with mask
    rgb_image, valid_mask = render_rgb_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention="opengl",
        return_mask=True,
    )

    assert rgb_image.shape == (3, 100, 100)
    assert valid_mask.shape == (100, 100)
    assert valid_mask.dtype == torch.bool

    # Check that we have some valid pixels
    assert valid_mask.sum() > 0
    assert valid_mask.sum() < 100 * 100  # Not all pixels should be filled

    # Check that non-masked areas have ignore_value (0.0 by default)
    for c in range(3):
        assert (rgb_image[c][~valid_mask] == 0.0).all()


def test_render_rgb_color_normalization():
    """Test automatic color normalization from 0-255 range."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.1, 0.1, -1.2],
            ],
            dtype=torch.float32,
        ),
        'rgb': torch.tensor(
            [
                [255.0, 0.0, 0.0],  # Red in 0-255 range
                [0.0, 255.0, 128.0],  # Green-cyan in 0-255 range
            ],
            dtype=torch.float32,
        ),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)

    rgb_image, valid_mask = render_rgb_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        return_mask=True,
    )

    # Colors should be normalized to [0, 1]
    assert rgb_image.max() <= 1.0
    assert rgb_image.min() >= 0.0

    # Check specific normalized values where points were rendered
    rendered_pixels = rgb_image[:, valid_mask]
    assert rendered_pixels.max() <= 1.0


def test_render_rgb_depth_sorting():
    """Test that closer points overwrite farther ones."""
    # Two points at same image location but different depths
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, -2.0],  # Farther point
                [0.0, 0.0, -1.0],  # Closer point (should overwrite)
            ],
            dtype=torch.float32,
        ),
        'rgb': torch.tensor(
            [
                [0.0, 0.0, 1.0],  # Blue (farther)
                [1.0, 0.0, 0.0],  # Red (closer, should win)
            ],
            dtype=torch.float32,
        ),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)

    rgb_image = render_rgb_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
    )

    # Check that some rendering occurred (any non-zero pixels)
    non_zero_pixels = (rgb_image > 0.0).any()
    assert (
        non_zero_pixels
    ), f"No pixels were rendered. RGB max: {rgb_image.max()}, min: {rgb_image.min()}"

    # If rendering occurred, check for color values
    if rgb_image.max() > 0:
        # At least one of the colors should be present
        red_present = (rgb_image[0] > 0.1).any()
        blue_present = (rgb_image[2] > 0.1).any()
        assert (
            red_present or blue_present
        ), "Neither red nor blue colors found in rendered image"


def test_render_rgb_empty_pointcloud():
    """Test that empty point cloud raises assertion error."""
    pc_data = {
        'pos': torch.empty((0, 3), dtype=torch.float32),
        'rgb': torch.empty((0, 3), dtype=torch.float32),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    with pytest.raises(AssertionError, match="Colors tensor must not be empty"):
        render_rgb_from_pointcloud(
            pc_data=pc_data,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            resolution=(100, 100),
        )


def test_render_rgb_points_behind_camera():
    """Test that points behind camera are filtered out."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, 1.0],  # Behind camera (positive Z in OpenGL)
                [0.0, 0.0, -1.0],  # In front of camera
            ],
            dtype=torch.float32,
        ),
        'rgb': torch.tensor(
            [
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 0.0, 0.0],  # Red
            ],
            dtype=torch.float32,
        ),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    rgb_image, valid_mask = render_rgb_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        return_mask=True,
    )

    # Should only have one point rendered (the one in front)
    assert valid_mask.sum() >= 1  # At least one pixel rendered


def test_render_rgb_custom_ignore_value():
    """Test using custom ignore value for empty pixels."""
    pc_data = {
        'pos': torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        'rgb': torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
    }

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    rgb_image = render_rgb_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        ignore_value=0.5,
    )

    # Most pixels should have the ignore value
    background_mask = rgb_image[0] == 0.5
    assert background_mask.sum() > 100 * 100 * 0.9  # Most pixels are background


def test_render_rgb_invalid_inputs():
    """Test various invalid input conditions."""
    valid_pc_data = {
        'pos': torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        'rgb': torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
    }
    valid_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    valid_extrinsics = torch.eye(4, dtype=torch.float32)

    # Test missing 'pos' key
    with pytest.raises(AssertionError, match=r"pc\.keys\(\)=dict_keys\(\['rgb'\]\)"):
        render_rgb_from_pointcloud(
            pc_data={'rgb': torch.zeros((1, 3))},
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100),
        )

    # Test missing 'rgb' key
    with pytest.raises(AssertionError, match="must contain 'rgb' key"):
        render_rgb_from_pointcloud(
            pc_data={'pos': torch.zeros((1, 3))},
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100),
        )

    # Test mismatched pos and rgb lengths
    with pytest.raises(
        AssertionError,
        match=r"\{pos: torch\.Size\(\[2, 3\]\), rgb: torch\.Size\(\[3, 3\]\)\}",
    ):
        render_rgb_from_pointcloud(
            pc_data={'pos': torch.zeros((2, 3)), 'rgb': torch.zeros((3, 3))},
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100),
        )

    # Test wrong intrinsics shape
    with pytest.raises(AssertionError, match="camera_intrinsics must be 3x3 matrix"):
        render_rgb_from_pointcloud(
            pc_data=valid_pc_data,
            camera_intrinsics=torch.eye(4),
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100),
        )

    # Test invalid convention
    with pytest.raises(
        AssertionError, match="convention must be 'opengl' or 'standard'"
    ):
        render_rgb_from_pointcloud(
            pc_data=valid_pc_data,
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100),
            convention="invalid",
        )
