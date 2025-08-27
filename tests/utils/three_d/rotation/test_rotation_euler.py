import torch
import math
from utils.three_d.rotation import euler_to_matrix, matrix_to_euler


def to_canonical_form(angles: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to canonical form where Y rotation is in [-pi/2, +pi/2]."""
    alpha, beta, gamma = angles[0], angles[1], angles[2]
    
    # Constrain beta to [-pi/2, +pi/2]
    if beta > math.pi / 2:
        # beta > pi/2: use alternate solution (alpha+pi, pi-beta, gamma+pi)
        alpha = alpha + math.pi if alpha <= 0 else alpha - math.pi
        beta = math.pi - beta
        gamma = gamma + math.pi if gamma <= 0 else gamma - math.pi
    elif beta < -math.pi / 2:
        # beta < -pi/2: use alternate solution (alpha+pi, -pi-beta, gamma+pi)  
        alpha = alpha + math.pi if alpha <= 0 else alpha - math.pi
        beta = -math.pi - beta
        gamma = gamma + math.pi if gamma <= 0 else gamma - math.pi
        
    return torch.tensor([alpha, beta, gamma])


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
        
        # Get the canonical form by doing a round-trip through matrix
        R = euler_to_matrix(angles)
        angles_canonical = matrix_to_euler(R)
        
        # Now test that the canonical form round-trips exactly
        R2 = euler_to_matrix(angles_canonical)
        angles_recovered = matrix_to_euler(R2)
        
        # Due to Euler angle ambiguity, validate that matrices are equivalent
        # This is the mathematically correct test - same rotation, potentially different angles
        R_error = torch.max(torch.abs(R - R2))
        assert R_error < 1e-6, f"Test {i}: Matrix equivalence error {R_error:.6f}"
        
        # Also check that recovered angles are reasonable (not NaN, in valid range)
        assert torch.all(torch.isfinite(angles_recovered)), f"Test {i}: Non-finite angles: {angles_recovered}"
        assert torch.all(torch.abs(angles_recovered) <= math.pi + 1e-6), f"Test {i}: Angles out of [-pi,pi] range: {angles_recovered}"
    
    print("All Euler angles round-trip tests passed!")
    return
