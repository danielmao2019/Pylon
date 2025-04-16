import pytest
import torch
import numpy as np
from scipy.spatial.transform import Rotation


from metrics.vision_3d import RegistrationRecall


def compute_rotation_translation_error_alt_numpy(estimated_transform, ground_truth_transform):
    """
    Alternative implementation using scipy.spatial.transform.Rotation
    to compute rotation and translation errors
    """
    # Extract rotation matrices and translation vectors
    R_est = estimated_transform[:3, :3]
    t_est = estimated_transform[:3, 3]

    R_gt = ground_truth_transform[:3, :3]
    t_gt = ground_truth_transform[:3, 3]

    # Alternative way: convert rotation matrices to scipy rotations
    # and use .inv() and multiplication to get the relative rotation
    rot_est = Rotation.from_matrix(R_est)
    rot_gt = Rotation.from_matrix(R_gt)

    # Get relative rotation (different approach than the implementation)
    # R_relative = R_gt^(-1) * R_est
    rot_relative = rot_gt.inv() * rot_est

    # Get angle in degrees using .magnitude() instead of trace formula
    angle_deg = rot_relative.magnitude() * 180.0 / np.pi

    # Translation error using L2 norm
    trans_error = np.linalg.norm(t_est - t_gt)

    return {
        "rotation_error_deg": angle_deg,
        "translation_error_m": trans_error
    }


def compute_registration_recall_alt_numpy(transforms_estimated, transforms_ground_truth,
                                     rot_threshold_deg=5.0, trans_threshold_m=0.3):
    """
    Alternative implementation to compute registration recall
    """
    if len(transforms_estimated) != len(transforms_ground_truth):
        raise ValueError("Number of estimated transforms must match number of ground truth transforms")

    successful_registrations = 0
    rotation_errors = []
    translation_errors = []

    for est_transform, gt_transform in zip(transforms_estimated, transforms_ground_truth):
        errors = compute_rotation_translation_error_alt_numpy(est_transform, gt_transform)

        rotation_errors.append(errors["rotation_error_deg"])
        translation_errors.append(errors["translation_error_m"])

        if (errors["rotation_error_deg"] < rot_threshold_deg and
            errors["translation_error_m"] < trans_threshold_m):
            successful_registrations += 1

    registration_recall = successful_registrations / len(transforms_estimated) if len(transforms_estimated) > 0 else 0

    return {
        "registration_recall": registration_recall,
        "avg_rotation_error_deg": np.mean(rotation_errors) if rotation_errors else 0,
        "avg_translation_error_m": np.mean(translation_errors) if translation_errors else 0
    }


def compute_rotation_error_trace_formula(R_pred, R_true):
    """Original trace formula used in PyTorch implementation."""
    R_error = np.matmul(R_true.T, R_pred)
    rot_trace = np.trace(R_error)
    rot_trace = min(3.0, max(-1.0, rot_trace))  # Clamp to avoid numerical issues
    angle_rad = np.arccos((rot_trace - 1) / 2)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def test_single_transform():
    # Create sample transformation matrices
    estimated_np = np.array([
        [0.9397, -0.3420, 0.0000, 0.1000],
        [0.3420, 0.9397, 0.0000, 0.2000],
        [0.0000, 0.0000, 1.0000, 0.3000],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ])  # This is a rotation of 20 degrees around z-axis

    ground_truth_np = np.array([
        [1.0000, 0.0000, 0.0000, 0.0800],
        [0.0000, 1.0000, 0.0000, 0.1900],
        [0.0000, 0.0000, 1.0000, 0.2800],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ])  # Identity rotation

    # Convert to PyTorch tensors
    estimated_torch = torch.tensor(estimated_np, dtype=torch.float32)
    ground_truth_torch = torch.tensor(ground_truth_np, dtype=torch.float32)

    # Compute registration recall metrics using PyTorch implementation
    metric = RegistrationRecall()
    torch_result = metric._compute_score(estimated_torch, ground_truth_torch)

    # Compute registration recall metrics using alternative NumPy implementation
    numpy_result = compute_rotation_translation_error_alt_numpy(estimated_np, ground_truth_np)

    # We use larger tolerance here because the two methods calculate rotation angles differently,
    # but they should be close for small angles
    assert abs(torch_result["rotation_error_deg"].item() - numpy_result["rotation_error_deg"]) < 1e-3, \
        f"PyTorch rotation error: {torch_result['rotation_error_deg'].item()}, NumPy rotation error: {numpy_result['rotation_error_deg']}"

    # Translation errors should match exactly
    assert abs(torch_result["translation_error_m"].item() - numpy_result["translation_error_m"]) < 1e-5, \
        f"PyTorch translation error: {torch_result['translation_error_m'].item()}, NumPy translation error: {numpy_result['translation_error_m']}"


def test_known_rotation_error():
    """Test with known rotation angles to verify correctness"""
    # Define rotations with known angles

    # Identity matrix (no rotation)
    R_identity = np.eye(3)

    # 30-degree rotation around Z axis
    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    R_z_30deg = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

    # Create full transformation matrices
    identity_transform = np.eye(4)
    identity_transform[:3, :3] = R_identity

    rotated_transform = np.eye(4)
    rotated_transform[:3, :3] = R_z_30deg

    # Convert to PyTorch tensors
    identity_torch = torch.tensor(identity_transform, dtype=torch.float32)
    rotated_torch = torch.tensor(rotated_transform, dtype=torch.float32)

    # Compute using PyTorch implementation
    metric = RegistrationRecall()
    torch_result = metric._compute_score(rotated_torch, identity_torch)

    # The rotation error should be 30 degrees
    expected_rotation_error = 30.0

    assert abs(torch_result["rotation_error_deg"].item() - expected_rotation_error) < 1e-3, \
        f"Expected rotation error of {expected_rotation_error} degrees, got {torch_result['rotation_error_deg'].item()}"


@pytest.mark.parametrize("angle_deg", [5.0, 10.0, 30.0, 60.0, 90.0, 180.0])
def test_various_rotation_angles(angle_deg):
    """Test rotation error calculation with various angles."""
    # Create rotation matrix around Z axis
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R_z_rotated = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

    # Identity rotation matrix
    R_identity = np.eye(3)

    # Create full transformation matrices
    rotated_transform = np.eye(4)
    rotated_transform[:3, :3] = R_z_rotated

    identity_transform = np.eye(4)
    identity_transform[:3, :3] = R_identity

    # Convert to PyTorch tensors
    rotated_torch = torch.tensor(rotated_transform, dtype=torch.float32)
    identity_torch = torch.tensor(identity_transform, dtype=torch.float32)

    # Compute using PyTorch implementation
    metric = RegistrationRecall()
    torch_result = metric._compute_score(rotated_torch, identity_torch)

    # Compute using trace formula (one formula implementation)
    trace_result = compute_rotation_error_trace_formula(R_z_rotated, R_identity)

    # Compute using scipy implementation (alternative implementation)
    scipy_result = compute_rotation_translation_error_alt_numpy(rotated_transform, identity_transform)["rotation_error_deg"]

    # The rotation error should match the input angle
    assert abs(torch_result["rotation_error_deg"].item() - angle_deg) < 1e-3, \
        f"Expected rotation error of {angle_deg} degrees, got {torch_result['rotation_error_deg'].item()}"

    # The trace formula and scipy implementations should give similar results
    assert abs(trace_result - scipy_result) < 1e-3, \
        f"Trace formula: {trace_result:.4f}°, Scipy: {scipy_result:.4f}°, Diff: {abs(trace_result - scipy_result):.4f}°"


def test_batch_transforms():
    # Create sample batched transformation matrices
    estimated_np = np.array([
        # First transform - 20 degree rotation around z-axis
        [
            [0.9397, -0.3420, 0.0000, 0.1000],
            [0.3420, 0.9397, 0.0000, 0.2000],
            [0.0000, 0.0000, 1.0000, 0.3000],
            [0.0000, 0.0000, 0.0000, 1.0000]
        ],
        # Second transform - 10 degree rotation around z-axis
        [
            [0.9848, -0.1736, 0.0000, 0.5000],
            [0.1736, 0.9848, 0.0000, 0.6000],
            [0.0000, 0.0000, 1.0000, 0.7000],
            [0.0000, 0.0000, 0.0000, 1.0000]
        ],
        # Third transform - 90 degree rotation around x-axis (very different)
        [
            [1.0000, 0.0000, 0.0000, 2.0000],
            [0.0000, 0.0000, -1.0000, 2.0000],
            [0.0000, 1.0000, 0.0000, 2.0000],
            [0.0000, 0.0000, 0.0000, 1.0000]
        ]
    ])

    ground_truth_np = np.array([
        # First transform - identity
        [
            [1.0000, 0.0000, 0.0000, 0.0800],
            [0.0000, 1.0000, 0.0000, 0.1900],
            [0.0000, 0.0000, 1.0000, 0.2800],
            [0.0000, 0.0000, 0.0000, 1.0000]
        ],
        # Second transform - 5 degree rotation around z-axis
        [
            [0.9962, -0.0872, 0.0000, 0.5200],
            [0.0872, 0.9962, 0.0000, 0.6200],
            [0.0000, 0.0000, 1.0000, 0.7200],
            [0.0000, 0.0000, 0.0000, 1.0000]
        ],
        # Third transform - identity
        [
            [1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000]
        ]
    ])

    # Expected rotation errors: 20 degrees, 5 degrees, 90 degrees
    expected_rotation_errors = np.array([20.0, 5.0, 90.0])

    # Convert to PyTorch tensors
    estimated_torch = torch.tensor(estimated_np, dtype=torch.float32)
    ground_truth_torch = torch.tensor(ground_truth_np, dtype=torch.float32)

    # Set thresholds
    rot_threshold_deg = 15.0  # This should make only the first transform fail the rotation check
    trans_threshold_m = 0.5  # This should make only the third transform fail the translation check

    # Compute registration recall metrics using PyTorch implementation
    metric = RegistrationRecall(rot_threshold_deg=rot_threshold_deg, trans_threshold_m=trans_threshold_m)
    torch_result = metric._compute_score(estimated_torch, ground_truth_torch)

    # Compute metrics for each transform using alternative NumPy implementation
    numpy_recalls = compute_registration_recall_alt_numpy(
        estimated_np, ground_truth_np,
        rot_threshold_deg=rot_threshold_deg, trans_threshold_m=trans_threshold_m
    )

    # Expected recall: only the second transform should be successful (1/3 = 0.333...)
    expected_recall = 1.0 / 3.0

    # Check that the registration recall is approximately equal
    assert abs(torch_result["registration_recall"].item() - expected_recall) < 1e-5, \
        f"Expected recall {expected_recall}, got {torch_result['registration_recall'].item()}"

    # The alternative numpy implementation should give the same recall since it's using same thresholds
    assert abs(numpy_recalls["registration_recall"] - expected_recall) < 1e-5, \
        f"Expected NumPy recall {expected_recall}, got {numpy_recalls['registration_recall']}"
