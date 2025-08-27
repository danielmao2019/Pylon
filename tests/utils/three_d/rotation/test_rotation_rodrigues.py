import torch
import math
from utils.three_d.rotation import axis_angle_to_matrix, matrix_to_axis_angle


def test_axis_angle_round_trip():
    """Test axis-angle to matrix conversion and back using Rodrigues' formula.
    
    This test randomly samples 10 pairs of rotation axes (unit vectors) and rotation 
    angles from [0, 2π). For each sampled rotation, converts the axis-angle representation 
    to a rotation matrix using axis_angle_to_matrix, then converts this matrix immediately 
    back to axis-angle representation using matrix_to_axis_angle. Checks that the axis 
    and angle match (almost) exactly the values started from.
    """
    torch.manual_seed(42)  # For reproducible tests
    
    for i in range(10):
        # Sample random rotation axis (unit vector)
        axis = torch.randn(3)
        axis = axis / torch.norm(axis)  # Normalize to unit vector
        
        # Sample random rotation angle from [0, 2π)
        angle = torch.rand(1) * 2 * math.pi
        angle = angle.squeeze()
        
        # Convert axis-angle to matrix
        R = axis_angle_to_matrix(axis, angle)
        
        # Convert matrix back to axis-angle
        axis_recovered, angle_recovered = matrix_to_axis_angle(R)
        
        # Check that the rotation matrix is valid (determinant = 1, orthogonal)
        det = torch.det(R)
        assert torch.abs(det - 1.0) < 1e-5, f"Test {i}: Determinant should be 1, got {det:.6f}"
        
        orthogonal_check = torch.mm(R, R.T) - torch.eye(3)
        assert torch.max(torch.abs(orthogonal_check)) < 1e-5, f"Test {i}: Matrix should be orthogonal"
        
        # Check axis recovery (note: axis can be negated, so check both directions)
        axis_error = torch.min(
            torch.norm(axis - axis_recovered),
            torch.norm(axis + axis_recovered)  # Check opposite direction too
        )
        assert axis_error < 1e-5, f"Test {i}: Axis recovery error {axis_error:.6f}, original: {axis}, recovered: {axis_recovered}"
        
        # Check angle recovery
        angle_error = torch.abs(angle - angle_recovered)
        # Handle wrap-around case (e.g., 0.1 vs 2π-0.1 should be close)
        angle_error = torch.min(angle_error, torch.abs(angle_error - 2*math.pi))
        assert angle_error < 1e-5, f"Test {i}: Angle recovery error {angle_error:.6f}, original: {angle:.6f}, recovered: {angle_recovered:.6f}"
    
    print("All axis-angle round-trip tests passed!")
    return
