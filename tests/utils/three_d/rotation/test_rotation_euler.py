import torch
import math
from utils.three_d.rotation import euler_to_matrix, matrix_to_euler


def test_euler_angles_round_trip():
    """Test Euler angles to matrix conversion and back using XYZ convention.
    
    This test randomly samples 10 sets of Euler angles from [-π, +π] for XYZ convention.
    Converts the Euler angles to a rotation matrix using euler_to_matrix, then converts 
    this matrix back to Euler angles using matrix_to_euler. Checks that the angles 
    match the values started from.
    """
    torch.manual_seed(42)  # For reproducible tests
    
    for i in range(10):
        # Sample random Euler angles from [-π, +π] for XYZ convention
        angles = (torch.rand(3) - 0.5) * 2 * math.pi
        
        # Convert Euler angles to matrix
        R = euler_to_matrix(angles)
        
        # Convert matrix back to Euler angles
        angles_recovered = matrix_to_euler(R)
        
        # Check angle recovery
        angle_errors = torch.abs(angles - angles_recovered)
        max_error = torch.max(angle_errors)
        assert max_error < 1e-4, f"Test {i}: Angle recovery error {max_error:.6f}, original: {angles}, recovered: {angles_recovered}"
    
    print("All Euler angles round-trip tests passed!")
    return
