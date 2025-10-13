import torch
import math
from utils.three_d.rotation.euler import (
    euler_to_matrix,
    matrix_to_euler,
    euler_canonical,
)


def test_euler_canonical_idempotent():
    """Test that euler_canonical is idempotent - applying it twice gives same result as once.

    This test randomly samples 10 sets of Euler angles from [-π, +π], applies
    the canonical form helper once and twice, and verifies the results are identical.
    """
    torch.manual_seed(42)  # For reproducible tests

    for i in range(10):
        # Sample random Euler angles from [-pi, +pi]
        angles = (torch.rand(3) - 0.5) * 2 * math.pi

        # Apply canonical form once
        canonical_once = euler_canonical(angles)

        # Apply canonical form twice
        canonical_twice = euler_canonical(canonical_once)

        # Check that applying once and twice gives same result
        angle_errors = torch.abs(canonical_once - canonical_twice)
        max_error = torch.max(angle_errors)
        assert (
            max_error < 1e-5
        ), f"Test {i}: Canonical idempotent error {max_error:.6f}, once: {canonical_once}, twice: {canonical_twice}"


def test_euler_angles_round_trip():
    """Test Euler angles to matrix conversion and back using XYZ convention.

    This test randomly samples 10 sets of Euler angles from [-π, +π] for XYZ convention.
    Converts the Euler angles to a rotation matrix using euler_to_matrix, then converts
    this matrix back to Euler angles using matrix_to_euler. Checks that the angles
    match the values started from.
    """
    torch.manual_seed(42)  # For reproducible tests

    for i in range(10):
        # Sample random Euler angles from [-pi, +pi] for XYZ convention
        angles = (torch.rand(3) - 0.5) * 2 * math.pi

        # Convert to canonical form using helper
        angles_canonical = euler_canonical(angles)

        # Convert canonical Euler angles to matrix
        R = euler_to_matrix(angles_canonical)

        # Convert matrix back to Euler angles
        angles_recovered = matrix_to_euler(R)

        # Check exact angle recovery (canonical form should round-trip exactly)
        angle_errors = torch.abs(angles_canonical - angles_recovered)
        max_error = torch.max(angle_errors)
        assert (
            max_error < 1e-5
        ), f"Test {i}: Angle recovery error {max_error:.6f}, canonical: {angles_canonical}, recovered: {angles_recovered}"


def test_euler_matrix_round_trip():
    """Test matrix recovery: angles -> matrix -> angles -> matrix.

    This test randomly samples 10 sets of Euler angles, converts to matrix,
    back to angles, then back to matrix again. Checks that the original and
    final matrices are identical.
    """
    torch.manual_seed(42)  # For reproducible tests

    for i in range(10):
        # Sample random Euler angles from [-pi, +pi] for XYZ convention
        angles = (torch.rand(3) - 0.5) * 2 * math.pi

        # Convert to canonical form using helper
        angles_canonical = euler_canonical(angles)

        # Convert canonical Euler angles to matrix
        R_original = euler_to_matrix(angles_canonical)

        # Convert matrix back to Euler angles
        angles_recovered = matrix_to_euler(R_original)

        # Convert recovered angles back to matrix
        R_recovered = euler_to_matrix(angles_recovered)

        # Check exact matrix recovery
        matrix_errors = torch.abs(R_original - R_recovered)
        max_error = torch.max(matrix_errors)
        assert max_error < 1e-5, f"Test {i}: Matrix recovery error {max_error:.6f}"
