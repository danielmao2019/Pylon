import torch
import pytest
from utils.point_cloud_ops.knn.knn import knn


@pytest.mark.parametrize("method", ["faiss", "pytorch3d", "torch", "scipy"])
def test_knn_k_only(method):
    """Test knn with k specified but no radius - should return k nearest neighbors."""
    # Create simple test data
    query_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    reference_points = torch.tensor(
        [
            [0.1, 0.0, 0.0],  # Closest
            [0.2, 0.0, 0.0],  # Second closest
            [0.3, 0.0, 0.0],  # Third closest
            [1.0, 0.0, 0.0],  # Farthest
        ],
        dtype=torch.float32,
    )

    k = 2

    # Test with k only (no radius)
    distances, indices = knn(
        query_points=query_points,
        reference_points=reference_points,
        k=k,
        method=method,
        return_distances=True,
    )

    # Should return exactly k neighbors
    assert distances.shape == (1, k), f"Expected shape (1, {k}), got {distances.shape}"
    assert indices.shape == (1, k), f"Expected shape (1, {k}), got {indices.shape}"

    # Should be sorted by distance
    assert distances[0, 0] < distances[0, 1], "Distances should be sorted"
    assert indices[0, 0] == 0, "First neighbor should be index 0"
    assert indices[0, 1] == 1, "Second neighbor should be index 1"
