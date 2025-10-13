import torch
import math
from utils.three_d.rotation.rodrigues import (
    rodrigues_to_matrix,
    matrix_to_rodrigues,
    rodrigues_canonical,
)


def test_rodrigues_canonical_idempotent():
    """Test that rodrigues_canonical is idempotent - applying it twice gives same result as once.

    This test randomly samples 10 pairs of Rodrigues representations, applies
    the canonical form helper once and twice, and verifies the results are identical.
    """
    torch.manual_seed(42)  # For reproducible tests

    for i in range(10):
        # Sample random rotation axis (unit vector)
        axis = torch.randn(3)
        axis = axis / torch.norm(axis)  # Normalize to unit vector

        # Sample random rotation angle from [-pi, +pi]
        angle = (torch.rand(1) - 0.5) * 2 * math.pi
        angle = angle.squeeze()

        # Apply canonical form once
        axis_once, angle_once = rodrigues_canonical(axis, angle)

        # Apply canonical form twice
        axis_twice, angle_twice = rodrigues_canonical(axis_once, angle_once)

        # Check that applying once and twice gives same result
        angle_error = torch.abs(angle_once - angle_twice)
        axis_error = torch.norm(axis_once - axis_twice)

        assert (
            angle_error < 1e-5
        ), f"Test {i}: Canonical angle idempotent error {angle_error:.6f}, once: {angle_once:.6f}, twice: {angle_twice:.6f}"
        assert (
            axis_error < 1e-5
        ), f"Test {i}: Canonical axis idempotent error {axis_error:.6f}"


def test_rodrigues_round_trip():
    """Test Rodrigues to matrix conversion and back using Rodrigues' formula.

    This test randomly samples 10 pairs of rotation axes (unit vectors) and rotation
    angles from [-π, +π]. For each sampled rotation, converts the Rodrigues representation
    to a rotation matrix using rodrigues_to_matrix, then converts this matrix immediately
    back to Rodrigues representation using matrix_to_rodrigues. Checks that the axis
    and angle match (almost) exactly the values started from.
    """
    torch.manual_seed(42)  # For reproducible tests

    for i in range(10):
        # Sample random rotation axis (unit vector)
        axis = torch.randn(3)
        axis = axis / torch.norm(axis)  # Normalize to unit vector

        # Sample random rotation angle from [-pi, +pi]
        angle = (torch.rand(1) - 0.5) * 2 * math.pi
        angle = angle.squeeze()

        # Convert to canonical form (non-negative angle)
        axis_canonical, angle_canonical = rodrigues_canonical(axis, angle)

        # Convert canonical Rodrigues to matrix
        R = rodrigues_to_matrix(axis_canonical, angle_canonical)

        # Convert matrix back to Rodrigues
        axis_recovered, angle_recovered = matrix_to_rodrigues(R)

        # Check that the rotation matrix is valid (determinant = 1, orthogonal)
        det = torch.det(R)
        assert (
            torch.abs(det - 1.0) < 1e-5
        ), f"Test {i}: Determinant should be 1, got {det:.6f}"

        orthogonal_check = torch.mm(R, R.T) - torch.eye(3)
        assert (
            torch.max(torch.abs(orthogonal_check)) < 1e-5
        ), f"Test {i}: Matrix should be orthogonal"

        # Check exact Rodrigues recovery (canonical form should round-trip exactly)
        angle_error = torch.abs(angle_canonical - angle_recovered)
        axis_error = torch.norm(axis_canonical - axis_recovered)

        assert (
            angle_error < 1e-5
        ), f"Test {i}: Angle recovery error {angle_error:.6f}, canonical: {angle_canonical:.6f}, recovered: {angle_recovered:.6f}"
        assert axis_error < 1e-5, f"Test {i}: Axis recovery error {axis_error:.6f}"


def test_rodrigues_matrix_round_trip():
    """Test matrix recovery: Rodrigues -> matrix -> Rodrigues -> matrix.

    This test randomly samples 10 pairs of rotation axes and angles, converts to matrix,
    back to Rodrigues, then back to matrix again. Checks that the original and
    final matrices are identical.
    """
    torch.manual_seed(42)  # For reproducible tests

    for i in range(10):
        # Sample random rotation axis (unit vector)
        axis = torch.randn(3)
        axis = axis / torch.norm(axis)  # Normalize to unit vector

        # Sample random rotation angle from [-pi, +pi]
        angle = (torch.rand(1) - 0.5) * 2 * math.pi
        angle = angle.squeeze()

        # Convert to canonical form (non-negative angle)
        axis_canonical, angle_canonical = rodrigues_canonical(axis, angle)

        # Convert canonical Rodrigues to matrix
        R_original = rodrigues_to_matrix(axis_canonical, angle_canonical)

        # Convert matrix back to Rodrigues
        axis_recovered, angle_recovered = matrix_to_rodrigues(R_original)

        # Convert recovered Rodrigues back to matrix
        R_recovered = rodrigues_to_matrix(axis_recovered, angle_recovered)

        # Check exact matrix recovery
        matrix_errors = torch.abs(R_original - R_recovered)
        max_error = torch.max(matrix_errors)
        assert max_error < 1e-5, f"Test {i}: Matrix recovery error {max_error:.6f}"
