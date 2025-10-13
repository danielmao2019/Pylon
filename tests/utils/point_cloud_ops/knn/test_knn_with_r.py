import torch
import pytest
from utils.point_cloud_ops.knn.knn import knn


# Test case 1: Both queries have 2 neighbors within radius
test_case_1 = {
    "query_points": torch.tensor(
        [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=torch.float32
    ),
    "reference_points": torch.tensor(
        [
            [
                -0.3,
                0.0,
                0.0,
            ],  # Within radius of query[0] only (dist=0.3<0.5 from q0, dist=0.9>0.5 from q1)
            [
                0.9,
                0.0,
                0.0,
            ],  # Within radius of query[1] only (dist=0.9>0.5 from q0, dist=0.3<0.5 from q1)
            [
                0.3,
                0.0,
                0.0,
            ],  # Within radius of both queries (dist=0.3<0.5 from q0, dist=0.3<0.5 from q1)
            [2.0, 0.0, 0.0],  # Within radius of neither query (dist=2.0>0.5 from both)
        ],
        dtype=torch.float32,
    ),
    "radius": 0.5,
    "expected_distances": torch.tensor(
        [
            [0.3, 0.3],  # Query 0: ref0 and ref2 within radius
            [0.3, 0.3],  # Query 1: ref1 and ref2 within radius
        ],
        dtype=torch.float32,
    ),
    "expected_indices": torch.tensor(
        [
            [0, 2],  # Query 0: ref[0] and ref[2] (tie-break by index)
            [1, 2],  # Query 1: ref[1] and ref[2] (tie-break by index)
        ],
        dtype=torch.long,
    ),
}

# Test case 2: Different numbers of neighbors (1 vs 3)
test_case_2 = {
    "query_points": torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
    ),
    "reference_points": torch.tensor(
        [
            [
                0.2,
                0.0,
                0.0,
            ],  # Within radius of query[0] only (dist=0.2<0.4 from q0, dist=0.8>0.4 from q1)
            [
                0.8,
                0.0,
                0.0,
            ],  # Within radius of query[1] only (dist=0.8>0.4 from q0, dist=0.2<0.4 from q1)
            [
                1.1,
                0.0,
                0.0,
            ],  # Within radius of query[1] only (dist=1.1>0.4 from q0, dist=0.1<0.4 from q1)
            [
                1.3,
                0.0,
                0.0,
            ],  # Within radius of query[1] only (dist=1.3>0.4 from q0, dist=0.3<0.4 from q1)
            [2.0, 0.0, 0.0],  # Outside radius of both
        ],
        dtype=torch.float32,
    ),
    "radius": 0.4,
    "expected_distances": torch.tensor(
        [
            [0.2, float('inf'), float('inf')],  # Query 0: only 1 neighbor, pad with inf
            [0.2, 0.1, 0.3],  # Query 1: 3 neighbors in original reference order
        ],
        dtype=torch.float32,
    ),
    "expected_indices": torch.tensor(
        [
            [0, -1, -1],  # Query 0: only ref[0], pad with -1
            [1, 2, 3],  # Query 1: ref[1], ref[2], ref[3] (original reference order)
        ],
        dtype=torch.long,
    ),
}


@pytest.mark.parametrize("method", ["faiss", "pytorch3d", "torch", "scipy"])
@pytest.mark.parametrize("test_case", [test_case_1, test_case_2])
def test_knn_radius_only(method, test_case):
    """Test knn with k=None but radius specified - should return all points within radius."""
    query_points = test_case["query_points"]
    reference_points = test_case["reference_points"]
    radius = test_case["radius"]
    expected_distances = test_case["expected_distances"]
    expected_indices = test_case["expected_indices"]

    # Test with radius only (k=None)
    distances, indices = knn(
        query_points=query_points,
        reference_points=reference_points,
        k=None,
        radius=radius,
        method=method,
        return_distances=True,
    )

    # Shape should be (n_queries, max_neighbors_found)
    assert (
        distances.shape == expected_distances.shape
    ), f"Distance shape mismatch: got {distances.shape}, expected {expected_distances.shape}"
    assert (
        indices.shape == expected_indices.shape
    ), f"Indices shape mismatch: got {indices.shape}, expected {expected_indices.shape}"

    # Check distances exactly
    assert torch.allclose(
        distances, expected_distances, equal_nan=True
    ), f"Distances mismatch:\nGot:\n{distances}\nExpected:\n{expected_distances}"

    # Check indices exactly
    assert torch.equal(
        indices, expected_indices
    ), f"Indices mismatch:\nGot:\n{indices}\nExpected:\n{expected_indices}"
