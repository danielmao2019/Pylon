import torch
import pytest
from utils.point_cloud_ops.knn import knn


@pytest.mark.parametrize("method", ["faiss", "pytorch3d", "torch", "scipy"])
def test_knn_k_larger_than_points(method):
    """Test knn with k larger than number of reference points - should pad with inf/-1."""
    query_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    reference_points = torch.tensor([
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
    ], dtype=torch.float32)
    
    k = 5  # More than available points
    
    distances, indices = knn(
        query_points=query_points,
        reference_points=reference_points,
        k=k,
        method=method,
        return_distances=True
    )
    
    # Should return exactly k positions
    assert distances.shape == (1, k), f"Expected shape (1, {k}), got {distances.shape}"
    assert indices.shape == (1, k), f"Expected shape (1, {k}), got {indices.shape}"
    
    # First 2 should be valid, rest should be inf/-1
    assert distances[0, 0] < float('inf'), "First distance should be finite"
    assert distances[0, 1] < float('inf'), "Second distance should be finite"
    assert distances[0, 2] == float('inf'), "Third distance should be inf"
    assert indices[0, 2] == -1, "Third index should be -1"


@pytest.mark.parametrize("method", ["faiss", "pytorch3d", "torch", "scipy"])
def test_knn_radius_and_k_insufficient(method):
    """Test knn with k and radius where fewer than k points are within radius."""
    query_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    reference_points = torch.tensor([
        [0.1, 0.0, 0.0],  # Within radius
        [1.0, 0.0, 0.0],  # Outside radius
        [2.0, 0.0, 0.0],  # Outside radius
    ], dtype=torch.float32)
    
    k = 3
    radius = 0.5  # Only 1 point within radius
    
    distances, indices = knn(
        query_points=query_points,
        reference_points=reference_points,
        k=k,
        radius=radius,
        method=method,
        return_distances=True
    )
    
    # Should return k positions but only first should be valid
    assert distances.shape == (1, k), f"Expected shape (1, {k}), got {distances.shape}"
    assert indices.shape == (1, k), f"Expected shape (1, {k}), got {indices.shape}"
    
    # First should be within radius, rest should be inf/-1
    assert distances[0, 0] < radius, "First neighbor should be within radius"
    assert distances[0, 1] == float('inf'), "Second should be inf (outside radius)"
    assert distances[0, 2] == float('inf'), "Third should be inf (outside radius)"
    assert indices[0, 1] == -1, "Second index should be -1"
    assert indices[0, 2] == -1, "Third index should be -1"
