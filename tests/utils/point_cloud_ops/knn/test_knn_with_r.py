import torch
import pytest
from utils.point_cloud_ops.knn import knn


@pytest.mark.parametrize("method", ["faiss", "pytorch3d", "torch", "scipy"])
def test_knn_radius_only(method):
    """Test knn with k=None but radius specified - should return all points within radius."""
    # Create systematic test data covering all 4 combinations:
    # Query 0: [0.0, 0.0, 0.0]
    # Query 1: [0.6, 0.0, 0.0] (distance = 0.6 between queries)
    query_points = torch.tensor([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=torch.float32)
    reference_points = torch.tensor([
        [-0.3, 0.0, 0.0], # Case 1: Within radius of query[0] only (dist=0.3<0.5 from q0, dist=0.9>0.5 from q1)
        [0.9, 0.0, 0.0],  # Case 2: Within radius of query[1] only (dist=0.9>0.5 from q0, dist=0.3<0.5 from q1)
        [0.3, 0.0, 0.0],  # Case 3: Within radius of both queries (dist=0.3<0.5 from q0, dist=0.3<0.5 from q1)
        [2.0, 0.0, 0.0],  # Case 4: Within radius of neither query (dist=2.0>0.5 from both)
    ], dtype=torch.float32)
    
    radius = 0.5
    
    # Test with radius only (k=None)
    distances, indices = knn(
        query_points=query_points,
        reference_points=reference_points,
        k=None,
        radius=radius,
        method=method,
        return_distances=True
    )
    
    # Should return all reference points as k (since k=None defaults to all)
    assert distances.shape == (2, 4), f"Expected shape (2, 4), got {distances.shape}"
    assert indices.shape == (2, 4), f"Expected shape (2, 4), got {indices.shape}"
    
    # Define expected result tensors
    # k=None with radius means k=4 (all ref points), sorted by distance, with radius filtering
    # Points outside radius get inf distance and -1 index
    
    # Manual distance calculations:
    # Query 0 [0.0, 0.0, 0.0] to refs: [0.3, 0.9, 0.3, 2.0] -> sorted: [0.3(ref0), 0.3(ref2), inf, inf]  
    # Query 1 [0.6, 0.0, 0.0] to refs: [0.9, 0.3, 0.3, 1.4] -> sorted: [0.3(ref1), 0.3(ref2), inf, inf]
    
    # Expected distances (sorted by distance, with radius filtering to inf)
    expected_distances = torch.tensor([
        [0.3, 0.3, float('inf'), float('inf')],  # Query 0: valid refs get actual distance, filtered get inf
        [0.3, 0.3, float('inf'), float('inf')]   # Query 1: valid refs get actual distance, filtered get inf
    ], dtype=torch.float32)
    
    # Expected indices (allowing for tie-breaking between equal distances)
    # Query 0: [0.3(ref0), 0.3(ref2), 0.9(ref1), 2.0(ref3)] with radius filtering
    # Query 1: [0.3(ref1), 0.3(ref2), 0.9(ref0), 1.4(ref3)] with radius filtering
    # Points outside radius (>0.5) get index -1
    
    expected_indices = torch.tensor([
        [0, 2, -1, -1],  # Query 0: ref[0] (idx 0 < 2), ref[2], filtered, filtered
        [1, 2, -1, -1]   # Query 1: ref[1] (idx 1 < 2), ref[2], filtered, filtered
    ], dtype=torch.long)
    
    # Check distances exactly
    assert torch.allclose(distances, expected_distances, equal_nan=True), f"Distances mismatch:\nGot:\n{distances}\nExpected:\n{expected_distances}"
    
    # Check indices exactly (tie-breaking by original index order)
    assert torch.equal(indices, expected_indices), f"Indices mismatch:\nGot:\n{indices}\nExpected:\n{expected_indices}"
