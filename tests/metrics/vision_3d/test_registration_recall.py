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


def compute_registration_recall_alt_numpy(estimated_transform, ground_truth_transform,
                                     rot_threshold_deg=5.0, trans_threshold_m=0.3):
    """
    Alternative implementation to compute registration recall for a single transform pair
    """
    errors = compute_rotation_translation_error_alt_numpy(estimated_transform, ground_truth_transform)

    # Determine if registration is successful
    success = (errors["rotation_error_deg"] < rot_threshold_deg and
              errors["translation_error_m"] < trans_threshold_m)

    return {
        "registration_recall": 1.0 if success else 0.0,
        "rotation_error_deg": errors["rotation_error_deg"],
        "translation_error_m": errors["translation_error_m"]
    }


def compute_rotation_error_trace_formula(R_pred, R_true):
    """Original trace formula used in PyTorch implementation."""
    R_error = np.matmul(R_true.T, R_pred)
    rot_trace = np.trace(R_error)
    rot_trace = min(3.0, max(-1.0, rot_trace))  # Clamp to avoid numerical issues
    angle_rad = np.arccos((rot_trace - 1) / 2)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


# ===== PART 1: TESTS WITH KNOWN ROTATION/TRANSLATION VALUES =====

def create_rotation_matrix(angle_deg, axis='z'):
    """Create a rotation matrix for a given angle in degrees and axis."""
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    if axis.lower() == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'z':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unknown axis: {axis}")


def create_transform_matrix(rotation_matrix, translation_vector):
    """Create a 4x4 transformation matrix from rotation and translation."""
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_vector
    return transform


@pytest.mark.parametrize("angle_deg,axis", [
    (30, 'z'),
    (45, 'x'),
    (60, 'y'),
    (90, 'z'),
    (180, 'x')
])
def test_known_rotation_angles(angle_deg, axis):
    """Test with known rotation angles to verify correctness."""
    # Create rotation matrix with known angle
    R_rotated = create_rotation_matrix(angle_deg, axis)

    # Identity rotation matrix
    R_identity = np.eye(3)

    # Create full transformation matrices
    rotated_transform = create_transform_matrix(R_rotated, np.zeros(3))
    identity_transform = create_transform_matrix(R_identity, np.zeros(3))

    # Convert to PyTorch tensors
    rotated_torch = torch.tensor(rotated_transform, dtype=torch.float32)
    identity_torch = torch.tensor(identity_transform, dtype=torch.float32)

    # Create RegistrationRecall instance
    registration_recall = RegistrationRecall()

    # Compute using the metric class
    metric_result = registration_recall(rotated_torch, identity_torch)

    # The rotation error should match the input angle
    assert abs(metric_result["rotation_error_deg"].item() - angle_deg) < 1e-3, \
        f"Expected rotation error of {angle_deg} degrees, got {metric_result['rotation_error_deg'].item()}"

    # Translation error should be zero
    assert abs(metric_result["translation_error_m"].item() - 0.0) < 1e-5, \
        f"Expected translation error of 0.0, got {metric_result['translation_error_m'].item()}"


@pytest.mark.parametrize("translation_vector", [
    [0.1, 0.0, 0.0],
    [0.0, 0.2, 0.0],
    [0.0, 0.0, 0.3],
    [0.1, 0.2, 0.3]
])
def test_known_translation_vectors(translation_vector):
    """Test with known translation vectors to verify correctness."""
    # Create identity rotation matrix
    R_identity = np.eye(3)

    # Create full transformation matrices
    translated_transform = create_transform_matrix(R_identity, translation_vector)
    identity_transform = create_transform_matrix(R_identity, np.zeros(3))

    # Convert to PyTorch tensors
    translated_torch = torch.tensor(translated_transform, dtype=torch.float32)
    identity_torch = torch.tensor(identity_transform, dtype=torch.float32)

    # Create RegistrationRecall instance
    registration_recall = RegistrationRecall()

    # Compute using the metric class
    metric_result = registration_recall(translated_torch, identity_torch)

    # The rotation error should be zero
    assert abs(metric_result["rotation_error_deg"].item() - 0.0) < 1e-5, \
        f"Expected rotation error of 0.0 degrees, got {metric_result['rotation_error_deg'].item()}"

    # Translation error should match the L2 norm of the translation vector
    expected_translation_error = np.linalg.norm(translation_vector)
    assert abs(metric_result["translation_error_m"].item() - expected_translation_error) < 1e-5, \
        f"Expected translation error of {expected_translation_error}, got {metric_result['translation_error_m'].item()}"


@pytest.mark.parametrize("angle_deg,translation_vector", [
    (30, [0.1, 0.0, 0.0]),
    (45, [0.0, 0.2, 0.0]),
    (60, [0.0, 0.0, 0.3]),
    (90, [0.1, 0.2, 0.3])
])
def test_known_rotation_and_translation(angle_deg, translation_vector):
    """Test with known rotation angles and translation vectors."""
    # Create rotation matrix with known angle around z-axis
    R_rotated = create_rotation_matrix(angle_deg, 'z')

    # Identity rotation matrix
    R_identity = np.eye(3)

    # Create full transformation matrices
    transformed = create_transform_matrix(R_rotated, translation_vector)
    identity_transform = create_transform_matrix(R_identity, np.zeros(3))

    # Convert to PyTorch tensors
    transformed_torch = torch.tensor(transformed, dtype=torch.float32)
    identity_torch = torch.tensor(identity_transform, dtype=torch.float32)

    # Create RegistrationRecall instance
    registration_recall = RegistrationRecall()

    # Compute using the metric class
    metric_result = registration_recall(transformed_torch, identity_torch)

    # The rotation error should match the input angle
    assert abs(metric_result["rotation_error_deg"].item() - angle_deg) < 1e-3, \
        f"Expected rotation error of {angle_deg} degrees, got {metric_result['rotation_error_deg'].item()}"

    # Translation error should match the L2 norm of the translation vector
    expected_translation_error = np.linalg.norm(translation_vector)
    assert abs(metric_result["translation_error_m"].item() - expected_translation_error) < 1e-5, \
        f"Expected translation error of {expected_translation_error}, got {metric_result['translation_error_m'].item()}"


@pytest.mark.parametrize("rot_threshold_deg,trans_threshold_m,angle_deg,translation_vector,expected_recall", [
    (5.0, 0.3, 3.0, [0.1, 0.0, 0.0], 1.0),    # Within both thresholds
    (5.0, 0.3, 10.0, [0.1, 0.0, 0.0], 0.0),   # Rotation exceeds threshold
    (5.0, 0.3, 3.0, [0.5, 0.0, 0.0], 0.0),    # Translation exceeds threshold
    (10.0, 0.5, 8.0, [0.4, 0.0, 0.0], 1.0),   # Within both thresholds
    (10.0, 0.5, 12.0, [0.6, 0.0, 0.0], 0.0)   # Both exceed thresholds
])
def test_registration_recall_with_known_values(rot_threshold_deg, trans_threshold_m,
                                              angle_deg, translation_vector, expected_recall):
    """Test registration recall with known values and thresholds."""
    # Create rotation matrix with known angle around z-axis
    R_rotated = create_rotation_matrix(angle_deg, 'z')

    # Identity rotation matrix
    R_identity = np.eye(3)

    # Create full transformation matrices
    transformed = create_transform_matrix(R_rotated, translation_vector)
    identity_transform = create_transform_matrix(R_identity, np.zeros(3))

    # Convert to PyTorch tensors
    transformed_torch = torch.tensor(transformed, dtype=torch.float32)
    identity_torch = torch.tensor(identity_transform, dtype=torch.float32)

    # Create RegistrationRecall instance with specified thresholds
    registration_recall = RegistrationRecall(rot_threshold_deg=rot_threshold_deg,
                                            trans_threshold_m=trans_threshold_m)

    # Compute registration recall using the metric class
    metric_result = registration_recall(transformed_torch, identity_torch)

    # Check that the registration recall matches the expected value
    assert abs(metric_result["registration_recall"].item() - expected_recall) < 1e-5, \
        f"Expected registration recall of {expected_recall}, got {metric_result['registration_recall'].item()}"


# ===== PART 2: TESTS WITH RANDOM TRANSFORMS =====

def generate_random_transform():
    """Generate a random 4x4 transformation matrix."""
    # Generate random rotation (using axis-angle representation)
    angle = np.random.uniform(0, 180)  # Random angle between 0 and 180 degrees
    axis = np.random.rand(3)
    axis = axis / np.linalg.norm(axis)  # Normalize to get a unit vector

    # Convert to rotation matrix
    rot = Rotation.from_rotvec(angle * axis * np.pi / 180)
    R = rot.as_matrix()

    # Generate random translation
    t = np.random.rand(3) * 0.5  # Random translation between 0 and 0.5 meters

    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t

    return transform


def test_single_random_transform():
    """Test with a single random transform."""
    # Generate random transforms
    estimated_transform = generate_random_transform()
    ground_truth_transform = generate_random_transform()

    # Convert to PyTorch tensors
    estimated_torch = torch.tensor(estimated_transform, dtype=torch.float32)
    ground_truth_torch = torch.tensor(ground_truth_transform, dtype=torch.float32)

    # Create RegistrationRecall instance
    registration_recall = RegistrationRecall()

    # Compute using the metric class
    metric_result = registration_recall(estimated_torch, ground_truth_torch)

    # Compute using alternative NumPy implementation
    numpy_result = compute_rotation_translation_error_alt_numpy(estimated_transform, ground_truth_transform)

    # Check that the results are approximately equal
    # We use larger tolerance here because the two methods calculate rotation angles differently
    assert abs(metric_result["rotation_error_deg"].item() - numpy_result["rotation_error_deg"]) < 1e-2, \
        f"Metric rotation error: {metric_result['rotation_error_deg'].item()}, NumPy rotation error: {numpy_result['rotation_error_deg']}"

    # Translation errors should match more closely
    assert abs(metric_result["translation_error_m"].item() - numpy_result["translation_error_m"]) < 1e-5, \
        f"Metric translation error: {metric_result['translation_error_m'].item()}, NumPy translation error: {numpy_result['translation_error_m']}"


def test_multiple_random_transforms():
    """Test with multiple random transforms, processing them one by one."""
    # Generate random transforms
    num_transforms = 5
    estimated_transforms = [generate_random_transform() for _ in range(num_transforms)]
    ground_truth_transforms = [generate_random_transform() for _ in range(num_transforms)]

    # Set thresholds
    rot_threshold_deg = 15.0
    trans_threshold_m = 0.5

    # Create RegistrationRecall instance
    registration_recall = RegistrationRecall(rot_threshold_deg=rot_threshold_deg,
                                            trans_threshold_m=trans_threshold_m)

    # Process each transform pair individually
    metric_results = []
    numpy_results = []

    for est_transform, gt_transform in zip(estimated_transforms, ground_truth_transforms):
        # Convert to PyTorch tensors
        est_torch = torch.tensor(est_transform, dtype=torch.float32)
        gt_torch = torch.tensor(gt_transform, dtype=torch.float32)

        # Compute using the metric class
        metric_result = registration_recall(est_torch, gt_torch)
        metric_results.append(metric_result)

        # Compute using alternative NumPy implementation
        numpy_result = compute_registration_recall_alt_numpy(
            est_transform, gt_transform,
            rot_threshold_deg=rot_threshold_deg, trans_threshold_m=trans_threshold_m
        )
        numpy_results.append(numpy_result)

    # Check that the results are approximately equal for each transform pair
    for i, (metric_result, numpy_result) in enumerate(zip(metric_results, numpy_results)):
        # Check rotation error
        assert abs(metric_result["rotation_error_deg"].item() - numpy_result["rotation_error_deg"]) < 1e-2, \
            f"Transform {i}: Metric rotation error: {metric_result['rotation_error_deg'].item()}, NumPy rotation error: {numpy_result['rotation_error_deg']}"

        # Check translation error
        assert abs(metric_result["translation_error_m"].item() - numpy_result["translation_error_m"]) < 1e-5, \
            f"Transform {i}: Metric translation error: {metric_result['translation_error_m'].item()}, NumPy translation error: {numpy_result['translation_error_m']}"

        # Check registration recall
        assert abs(metric_result["registration_recall"].item() - numpy_result["registration_recall"]) < 1e-5, \
            f"Transform {i}: Metric registration recall: {metric_result['registration_recall'].item()}, NumPy registration recall: {numpy_result['registration_recall']}"


def test_registration_recall_edge_cases():
    """Test registration recall with edge cases."""
    # Test with identity transforms (perfect match)
    identity_transform = np.eye(4)
    identity_torch = torch.tensor(identity_transform, dtype=torch.float32)

    # Create RegistrationRecall instance
    registration_recall = RegistrationRecall()

    # Compute using the metric class
    metric_result = registration_recall(identity_torch, identity_torch)

    # Check that the results are as expected
    assert abs(metric_result["rotation_error_deg"].item() - 0.0) < 1e-5, \
        f"Expected rotation error of 0.0 degrees, got {metric_result['rotation_error_deg'].item()}"

    assert abs(metric_result["translation_error_m"].item() - 0.0) < 1e-5, \
        f"Expected translation error of 0.0, got {metric_result['translation_error_m'].item()}"

    assert abs(metric_result["registration_recall"].item() - 1.0) < 1e-5, \
        f"Expected registration recall of 1.0, got {metric_result['registration_recall'].item()}"
