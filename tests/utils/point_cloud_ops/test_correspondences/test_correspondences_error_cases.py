
import pytest
import torch
from utils.point_cloud_ops.correspondences import get_correspondences


def test_get_correspondences_invalid_dict_missing_pos():
    """Test that error is raised when dictionary doesn't have 'pos' key."""
    # Dictionary without 'pos' key
    invalid_src_dict = {
        'positions': torch.tensor([  # Wrong key name
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float32)
    }
    
    tgt_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    radius = 1.0
    
    # Should raise AssertionError due to check_point_cloud validation
    # The actual error message is: pc.keys()=dict_keys(['positions'])
    with pytest.raises(AssertionError, match=r"pc\.keys\(\)="):
        get_correspondences(invalid_src_dict, tgt_points, None, radius)


def test_get_correspondences_invalid_dict_wrong_pos_shape():
    """Test that error is raised when 'pos' has wrong shape."""
    # Dictionary with wrong shape for 'pos' 
    invalid_src_dict = {
        'pos': torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)  # 1D instead of 2D
    }
    
    tgt_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    radius = 1.0
    
    # Should raise AssertionError due to check_point_cloud validation
    # The actual error message is: Expected 2D tensor, got shape torch.Size([3])
    with pytest.raises(AssertionError, match="Expected 2D tensor"):
        get_correspondences(invalid_src_dict, tgt_points, None, radius)


def test_get_correspondences_mismatched_devices():
    """Test that error is raised when source and target are on different devices."""
    src_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)  # CPU tensor
    
    # This test can only run if CUDA is available
    if torch.cuda.is_available():
        tgt_points = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float32, device='cuda')  # GPU tensor
        
        radius = 1.0
        
        with pytest.raises(AssertionError, match="src_points.device="):
            get_correspondences(src_points, tgt_points, None, radius)


def test_get_correspondences_invalid_transform_shape():
    """Test that error is raised for invalid transform shape."""
    src_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    tgt_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    # Invalid transform shape (should be 4x4)
    invalid_transform = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    radius = 1.0
    
    with pytest.raises(AssertionError, match="Invalid transform shape"):
        get_correspondences(src_points, tgt_points, invalid_transform, radius)


def test_get_correspondences_negative_radius():
    """Test that error is raised for negative radius."""
    src_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    tgt_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    negative_radius = -0.5
    
    with pytest.raises(AssertionError, match="radius must be positive number"):
        get_correspondences(src_points, tgt_points, None, negative_radius)


def test_get_correspondences_zero_radius():
    """Test that error is raised for zero radius."""
    src_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    tgt_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    zero_radius = 0.0
    
    with pytest.raises(AssertionError, match="radius must be positive number"):
        get_correspondences(src_points, tgt_points, None, zero_radius)
