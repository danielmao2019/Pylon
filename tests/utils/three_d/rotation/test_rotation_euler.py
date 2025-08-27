import torch
import math
from utils.three_d.rotation import euler_to_matrix, matrix_to_euler


def test_euler_angles_round_trip():
    """Test Euler angles to matrix conversion and back using XYZ convention.
    
    This test randomly samples 10 sets of Euler angles from [0, 2π) for XYZ convention.
    For each sampled set of angles, converts the Euler angles to a rotation matrix using 
    euler_to_matrix, then converts this matrix immediately back to Euler angles using 
    matrix_to_euler. Checks that the angles match (almost) exactly the values started from.
    
    Note: This implementation uses XYZ convention (not ZYX as mentioned in comments).
    """
    torch.manual_seed(42)  # For reproducible tests
    
    for i in range(10):
        # Sample random Euler angles from [0, 2π) for XYZ convention
        # angles[0] = X rotation (alpha)
        # angles[1] = Y rotation (beta) 
        # angles[2] = Z rotation (gamma)
        angles = torch.rand(3) * 2 * math.pi
        
        # Convert Euler angles to matrix
        R = euler_to_matrix(angles)
        
        # Convert matrix back to Euler angles
        angles_recovered = matrix_to_euler(R)
        
        # Check that the rotation matrix is valid (determinant = 1, orthogonal)
        det = torch.det(R)
        assert torch.abs(det - 1.0) < 1e-5, f"Test {i}: Determinant should be 1, got {det:.6f}"
        
        orthogonal_check = torch.mm(R, R.T) - torch.eye(3)
        assert torch.max(torch.abs(orthogonal_check)) < 1e-5, f"Test {i}: Matrix should be orthogonal"
        
        # Check angle recovery
        # Handle potential gimbal lock and angle wrapping issues
        angle_errors = torch.abs(angles - angles_recovered)
        
        # Handle wrap-around case (e.g., 0.1 vs 2π-0.1 should be close)
        for j in range(3):
            angle_errors[j] = torch.min(angle_errors[j], torch.abs(angle_errors[j] - 2*math.pi))
        
        # Check that all angle errors are small
        max_error = torch.max(angle_errors)
        assert max_error < 1e-4, f"Test {i}: Angle recovery error {max_error:.6f}, original: {angles}, recovered: {angles_recovered}"
        
        # Additional test: verify the recovered rotation matrix produces the same result
        R_recovered = euler_to_matrix(angles_recovered)
        matrix_error = torch.max(torch.abs(R - R_recovered))
        assert matrix_error < 1e-5, f"Test {i}: Matrix reconstruction error {matrix_error:.6f}"
    
    print("All Euler angles round-trip tests passed!")
    return
