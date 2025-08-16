import pytest
import torch
from data.transforms.vision_3d.clamp import Clamp


def test_clamp_invalid_initialization():
    """Test Clamp transform initialization with invalid parameters."""
    # Test invalid max_points type
    with pytest.raises(AssertionError):
        Clamp(max_points="invalid")
    
    with pytest.raises(AssertionError):
        Clamp(max_points=1.5)
    
    with pytest.raises(AssertionError):
        Clamp(max_points=None)
    
    # Test invalid max_points values
    with pytest.raises(AssertionError):
        Clamp(max_points=0)
    
    with pytest.raises(AssertionError):
        Clamp(max_points=-1)
    
    with pytest.raises(AssertionError):
        Clamp(max_points=-100)


def test_clamp_invalid_input_types():
    """Test Clamp transform with invalid input types."""
    clamp = Clamp(max_points=100)
    
    # Test with non-dictionary input
    with pytest.raises(AssertionError, match="Argument 0 must be dict"):
        clamp("not a dict")
    
    with pytest.raises(AssertionError, match="Argument 0 must be dict"):
        clamp(123)
    
    with pytest.raises(AssertionError, match="Argument 0 must be dict"):
        clamp([1, 2, 3])
    
    # Test with no arguments
    with pytest.raises(AssertionError, match="len\\(args\\)=0"):
        clamp()


def test_clamp_invalid_point_cloud_structure():
    """Test Clamp transform with invalid point cloud structure."""
    clamp = Clamp(max_points=100)
    
    # Test with missing 'pos' key
    with pytest.raises(AssertionError, match="pc\\.keys\\(\\)"):
        clamp({'feat': torch.randn(10, 4, dtype=torch.float32)})
    
    # Test with non-tensor 'pos' value
    with pytest.raises(AssertionError, match="pc\\['pos'\\]"):
        clamp({'pos': "not a tensor"})
    
    # Test with non-tensor other values
    with pytest.raises(AssertionError, match="pc\\['feat'\\]"):
        clamp({
            'pos': torch.randn(10, 3, dtype=torch.float32),
            'feat': "not a tensor"
        })


def test_clamp_inconsistent_point_counts():
    """Test Clamp transform with point clouds having different point counts."""
    clamp = Clamp(max_points=50)
    
    pc1 = {
        'pos': torch.randn(100, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.randn(100, 4, dtype=torch.float32, device='cuda'),
    }
    pc2 = {
        'pos': torch.randn(200, 3, dtype=torch.float32, device='cuda'),  # Different count
        'feat': torch.randn(200, 4, dtype=torch.float32, device='cuda'),
    }
    
    with pytest.raises(ValueError, match="All point clouds must have the same number of points"):
        clamp(pc1, pc2, seed=42)


def test_clamp_inconsistent_devices():
    """Test Clamp transform with point clouds on different devices."""
    clamp = Clamp(max_points=50)
    
    pc1 = {
        'pos': torch.randn(100, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.randn(100, 4, dtype=torch.float32, device='cuda'),
    }
    pc2 = {
        'pos': torch.randn(100, 3, dtype=torch.float32, device='cpu'),  # Different device
        'feat': torch.randn(100, 4, dtype=torch.float32, device='cpu'),
    }
    
    with pytest.raises(ValueError, match="All point clouds must be on the same device"):
        clamp(pc1, pc2, seed=42)


def test_clamp_empty_point_cloud():
    """Test Clamp transform with empty point cloud."""
    clamp = Clamp(max_points=100)
    
    pc = {
        'pos': torch.empty(0, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.empty(0, 4, dtype=torch.float32, device='cuda'),
    }
    
    with pytest.raises(AssertionError, match="pc\\['pos'\\]\\.shape"):
        clamp(pc, seed=42)


def test_clamp_wrong_pos_dimensions():
    """Test Clamp transform with wrong 'pos' tensor dimensions."""
    clamp = Clamp(max_points=100)
    
    # Wrong number of dimensions (1D instead of 2D)
    with pytest.raises(AssertionError):
        clamp({
            'pos': torch.randn(10, dtype=torch.float32),  # 1D instead of 2D
            'feat': torch.randn(10, 4, dtype=torch.float32),
        })
    
    # Wrong last dimension (not 3)
    with pytest.raises(AssertionError):
        clamp({
            'pos': torch.randn(10, 2, dtype=torch.float32),  # Shape (10, 2) instead of (10, 3)
            'feat': torch.randn(10, 4, dtype=torch.float32),
        })


def test_clamp_inconsistent_shapes_within_pc():
    """Test Clamp transform with inconsistent shapes within single point cloud."""
    clamp = Clamp(max_points=100)
    
    # Different number of points in pos vs feat
    with pytest.raises(AssertionError):
        clamp({
            'pos': torch.randn(10, 3, dtype=torch.float32),
            'feat': torch.randn(8, 4, dtype=torch.float32),  # Different number of points
        })


def test_clamp_wrong_tensor_type():
    """Test Clamp transform with non-tensor values."""
    clamp = Clamp(max_points=100)
    
    # Non-tensor for 'pos'
    with pytest.raises(AssertionError):
        clamp({
            'pos': [[1, 2, 3], [4, 5, 6]],  # List instead of tensor
            'feat': torch.randn(2, 4, dtype=torch.float32),
        })


@pytest.mark.parametrize("invalid_input", [
    None,
    [],
    {},
    42,
    "string",
    torch.tensor([1, 2, 3]),
])
def test_clamp_various_invalid_inputs(invalid_input):
    """Test Clamp transform with various invalid input types."""
    clamp = Clamp(max_points=100)
    
    with pytest.raises(AssertionError):
        clamp(invalid_input)


def test_clamp_mixed_valid_invalid_multi_args():
    """Test multi-arg Clamp with mix of valid and invalid point clouds."""
    clamp = Clamp(max_points=50)
    
    valid_pc = {
        'pos': torch.randn(100, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.randn(100, 4, dtype=torch.float32, device='cuda'),
    }
    
    invalid_pc = {
        'pos': torch.randn(100, 2, dtype=torch.float32, device='cuda'),  # Wrong shape
        'feat': torch.randn(100, 4, dtype=torch.float32, device='cuda'),
    }
    
    with pytest.raises(AssertionError):
        clamp(valid_pc, invalid_pc, seed=42)


def test_clamp_multi_args_one_empty():
    """Test multi-arg Clamp where one point cloud is empty."""
    clamp = Clamp(max_points=50)
    
    valid_pc = {
        'pos': torch.randn(100, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.randn(100, 4, dtype=torch.float32, device='cuda'),
    }
    
    empty_pc = {
        'pos': torch.empty(0, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.empty(0, 4, dtype=torch.float32, device='cuda'),
    }
    
    # This should fail due to different point counts (100 vs 0)
    with pytest.raises(ValueError, match="All point clouds must have the same number of points"):
        clamp(valid_pc, empty_pc, seed=42)