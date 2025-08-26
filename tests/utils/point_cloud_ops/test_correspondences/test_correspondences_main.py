import torch
from utils.point_cloud_ops.correspondences import get_correspondences


def test_get_correspondences_basic():
    """Test basic correspondence finding functionality."""
    # Create simple test data
    src_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    tgt_points = torch.tensor([
        [0.1, 0.0, 0.0],  # Close to src[0]
        [1.1, 0.0, 0.0],  # Close to src[1]
        [5.0, 0.0, 0.0],  # Far from all src points
    ], dtype=torch.float32)
    
    radius = 0.5
    
    # Test without transform
    correspondences = get_correspondences(src_points, tgt_points, None, radius)
    
    # Should be [K, 2] format with [src_idx, tgt_idx] pairs
    assert correspondences.ndim == 2, "Correspondences should be 2D"
    assert correspondences.shape[1] == 2, "Each correspondence should have 2 indices"
    
    # Should find 2 correspondences (src[0]↔tgt[0], src[1]↔tgt[1])
    assert correspondences.shape[0] == 2, f"Expected 2 correspondences, got {correspondences.shape[0]}"
    
    # Check the actual correspondences
    corr_set = set((c[0].item(), c[1].item()) for c in correspondences)
    expected = {(0, 0), (1, 1)}  # src[0]↔tgt[0], src[1]↔tgt[1]
    assert corr_set == expected, f"Expected {expected}, got {corr_set}"


def test_get_correspondences_with_transform():
    """Test correspondence finding with transformation."""
    src_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    tgt_points = torch.tensor([
        [1.1, 0.0, 0.0],  # Will match src[0] after translation
        [2.1, 0.0, 0.0],  # Will match src[1] after translation
    ], dtype=torch.float32)
    
    # Translation transform: move src points by [1, 0, 0]
    transform = torch.tensor([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    radius = 0.2
    
    correspondences = get_correspondences(src_points, tgt_points, transform, radius)
    
    # Should find 2 correspondences after transformation
    assert correspondences.shape[0] == 2, f"Expected 2 correspondences, got {correspondences.shape[0]}"
    assert correspondences.shape[1] == 2, "Each correspondence should have 2 indices"
