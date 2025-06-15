import pytest
import torch
import numpy as np
from metrics.vision_3d.point_cloud_registration.isotropic_transform_error import IsotropicTransformError


def create_rotation_matrix(angle_deg: float, axis: str = 'z') -> torch.Tensor:
    """Create a 3x3 rotation matrix for a given angle around a specified axis."""
    angle_rad = np.radians(angle_deg)
    if axis == 'z':
        return torch.tensor([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
    elif axis == 'y':
        return torch.tensor([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ], dtype=torch.float32)
    elif axis == 'x':
        return torch.tensor([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ], dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def create_transform_matrix(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Create a 4x4 transformation matrix from rotation and translation."""
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


@pytest.fixture
def metric():
    return IsotropicTransformError()


@pytest.mark.parametrize("angle,axis,translation", [
    (45, 'z', [1.0, 2.0, 3.0]),
    (90, 'y', [0.0, 1.0, 0.0]),
    (180, 'x', [-1.0, -2.0, -3.0]),
])
def test_rotation_translation_decomposition(metric, angle, axis, translation):
    """Test the decomposition of transformation matrix into rotation and translation."""
    # Create a test transformation matrix
    rotation = create_rotation_matrix(angle, axis)
    translation = torch.tensor(translation, dtype=torch.float32)
    transform = create_transform_matrix(rotation, translation)

    # Test decomposition
    decomposed_rot, decomposed_trans = metric._get_rotation_translation(transform)

    # Check results
    assert torch.allclose(decomposed_rot, rotation)
    assert torch.allclose(decomposed_trans, translation)


@pytest.mark.parametrize("angle_gt,angle_pred,expected_error", [
    (0, 0, 0),      # No rotation
    (45, 45, 0),    # Same rotation
    (45, -45, 90),  # Opposite rotation
    (90, 0, 90),    # Quarter turn difference
    (180, 0, 180),  # Half turn difference
])
def test_rotation_error_computation(metric, angle_gt, angle_pred, expected_error):
    """Test rotation error computation for various angles."""
    # Create rotation matrices
    rot_gt = create_rotation_matrix(angle_gt, 'z')
    rot_pred = create_rotation_matrix(angle_pred, 'z')

    # Compute error
    error = metric._compute_rotation_error(rot_gt, rot_pred)

    # Check result
    assert torch.allclose(error, torch.tensor(expected_error, dtype=torch.float32), atol=1e-5)


@pytest.mark.parametrize("trans_gt,trans_pred,expected_error", [
    ([0., 0., 0.], [0., 0., 0.], 0.),           # No translation
    ([1., 0., 0.], [1., 0., 0.], 0.),           # Same translation
    ([1., 0., 0.], [0., 0., 0.], 1.),           # Unit difference
    ([1., 1., 1.], [0., 0., 0.], np.sqrt(3)),   # Diagonal difference
])
def test_translation_error_computation(metric, trans_gt, trans_pred, expected_error):
    """Test translation error computation."""
    # Convert lists to tensors
    trans_gt = torch.tensor(trans_gt, dtype=torch.float32)
    trans_pred = torch.tensor(trans_pred, dtype=torch.float32)

    # Compute error
    error = metric._compute_translation_error(trans_gt, trans_pred)

    # Check result
    assert torch.allclose(error, torch.tensor(expected_error, dtype=torch.float32), atol=1e-5)


@pytest.mark.parametrize("angle_gt,angle_pred,trans_gt,trans_pred,expected_rre,expected_rte", [
    (45, 50, [1.0, 2.0, 3.0], [1.1, 2.1, 3.1], 5.0, np.sqrt(0.03)),  # Small differences
    (0, 90, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 90.0, np.sqrt(3)),      # Large differences
    (180, 180, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.0, 0.0),           # No differences
])
def test_complete_transform_error(metric, angle_gt, angle_pred, trans_gt, trans_pred, expected_rre, expected_rte):
    """Test complete transform error computation with both rotation and translation."""
    # Create test transformation matrices
    rot_gt = create_rotation_matrix(angle_gt, 'z')
    trans_gt = torch.tensor(trans_gt, dtype=torch.float32)
    transform_gt = create_transform_matrix(rot_gt, trans_gt)

    # Create predicted transformation
    rot_pred = create_rotation_matrix(angle_pred, 'z')
    trans_pred = torch.tensor(trans_pred, dtype=torch.float32)
    transform_pred = create_transform_matrix(rot_pred, trans_pred)

    # Compute errors
    scores = metric(
        y_pred={'transform': transform_pred.unsqueeze(0)},
        y_true={'transform': transform_gt.unsqueeze(0)}
    )

    # Check results
    assert 'RRE' in scores
    assert 'RTE' in scores
    assert torch.allclose(scores['RRE'], torch.tensor(expected_rre, dtype=torch.float32), atol=1e-5)
    assert torch.allclose(scores['RTE'], torch.tensor(expected_rte, dtype=torch.float32), atol=1e-5)


@pytest.mark.parametrize("y_pred,y_true", [
    (torch.eye(4), {'transform': torch.eye(4)}),
    ({'transform': torch.eye(4)}, torch.eye(4)),
    ({'wrong_key': torch.eye(4)}, {'transform': torch.eye(4)}),
    ({'transform': torch.eye(4)}, {'wrong_key': torch.eye(4)}),
])
def test_input_edge_cases(metric, y_pred, y_true):
    """Test input validation for various edge cases."""
    metric(y_pred=y_pred, y_true=y_true)


@pytest.mark.parametrize("y_pred,y_true", [
    ({'transform': torch.eye(3)}, {'transform': torch.eye(4)}),
])
def test_invalid_inputs(metric, y_pred, y_true):
    """Test invalid inputs."""
    with pytest.raises(AssertionError):
        metric(y_pred=y_pred, y_true=y_true)
