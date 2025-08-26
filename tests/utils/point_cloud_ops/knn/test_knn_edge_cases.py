import torch
import pytest
from utils.point_cloud_ops.knn.knn import knn


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
def test_knn_radius_insufficient_points(method):
    """Test knn with radius where some queries have no points within radius."""
    query_points = torch.tensor([
        [0.0, 0.0, 0.0],   # Has points within radius
        [10.0, 0.0, 0.0],  # No points within radius
    ], dtype=torch.float32)
    reference_points = torch.tensor([
        [0.1, 0.0, 0.0],  # Within radius of first query only
        [0.2, 0.0, 0.0],  # Within radius of first query only
        [0.3, 0.0, 0.0],  # Within radius of first query only
    ], dtype=torch.float32)
    
    radius = 0.5
    
    distances, indices = knn(
        query_points=query_points,
        reference_points=reference_points,
        k=None,
        radius=radius,
        method=method,
        return_distances=True
    )
    
    # Expected shape: (2 queries, 3 max_neighbors)
    # First query should find 3 neighbors within radius, second query should find 0
    assert distances.shape == (2, 3), f"Expected shape (2, 3), got {distances.shape}"
    assert indices.shape == (2, 3), f"Expected shape (2, 3), got {indices.shape}"
    
    # Expected distances:
    # Query 0 [0.0, 0.0, 0.0] to refs:
    #   - ref0 [0.1, 0.0, 0.0]: dist = 0.1 (within radius 0.5)
    #   - ref1 [0.2, 0.0, 0.0]: dist = 0.2 (within radius 0.5)
    #   - ref2 [0.3, 0.0, 0.0]: dist = 0.3 (within radius 0.5)
    # Query 1 [10.0, 0.0, 0.0] to refs:
    #   - ref0 [0.1, 0.0, 0.0]: dist = 9.9 (outside radius 0.5)
    #   - ref1 [0.2, 0.0, 0.0]: dist = 9.8 (outside radius 0.5)  
    #   - ref2 [0.3, 0.0, 0.0]: dist = 9.7 (outside radius 0.5)
    
    expected_distances = torch.tensor([
        [0.1, 0.2, 0.3],  # Query 0: all 3 within radius, sorted by distance
        [float('inf'), float('inf'), float('inf')]  # Query 1: none within radius
    ], dtype=torch.float32)
    
    expected_indices = torch.tensor([
        [0, 1, 2],  # Query 0: indices 0, 1, 2 (sorted by distance)
        [-1, -1, -1]  # Query 1: all -1 (no valid neighbors)
    ], dtype=torch.long)
    
    # Check distances exactly
    assert torch.allclose(distances, expected_distances, equal_nan=True), f"Distances mismatch:\nGot:\n{distances}\nExpected:\n{expected_distances}"
    
    # Check indices exactly
    assert torch.equal(indices, expected_indices), f"Indices mismatch:\nGot:\n{indices}\nExpected:\n{expected_indices}"
