import torch
import pytest
from utils.point_cloud_ops.knn import knn


@pytest.mark.parametrize("method", ["faiss", "pytorch3d", "torch", "scipy"])
def test_knn_radius_and_k(method):
    """Test knn with both k and radius specified - should return k nearest within radius."""
    # Create simple test data
    query_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    reference_points = torch.tensor([
        [0.1, 0.0, 0.0],  # Within radius
        [0.2, 0.0, 0.0],  # Within radius  
        [0.3, 0.0, 0.0],  # Within radius
        [1.0, 0.0, 0.0],  # Outside radius
    ], dtype=torch.float32)
    
    k = 2
    radius = 0.5
    
    # Test with both k and radius
    distances, indices = knn(
        query_points=query_points,
        reference_points=reference_points,
        k=k,
        radius=radius,
        method=method,
        return_distances=True
    )
    
    # Should return exactly k positions
    assert distances.shape == (1, k), f"Expected shape (1, {k}), got {distances.shape}"
    assert indices.shape == (1, k), f"Expected shape (1, {k}), got {indices.shape}"
    
    # Both should be within radius (closest 2 of the 3 within radius)
    assert distances[0, 0] < radius, "First neighbor should be within radius"
    assert distances[0, 1] < radius, "Second neighbor should be within radius"
    assert indices[0, 0] == 0, "First neighbor should be closest"
    assert indices[0, 1] == 1, "Second neighbor should be second closest"
