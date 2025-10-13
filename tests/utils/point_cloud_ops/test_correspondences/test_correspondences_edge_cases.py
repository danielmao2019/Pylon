import torch
from utils.point_cloud_ops.correspondences import get_correspondences


def test_get_correspondences_empty():
    """Test correspondence finding when no matches within radius."""
    src_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    tgt_points = torch.tensor(
        [
            [10.0, 0.0, 0.0],  # Far from all src points
            [20.0, 0.0, 0.0],  # Far from all src points
        ],
        dtype=torch.float32,
    )

    radius = 0.5

    correspondences = get_correspondences(src_points, tgt_points, None, radius)

    # Should return empty tensor with correct shape
    assert correspondences.shape == (
        0,
        2,
    ), f"Expected (0, 2), got {correspondences.shape}"
