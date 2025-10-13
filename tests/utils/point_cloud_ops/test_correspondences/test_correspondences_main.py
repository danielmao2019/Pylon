import torch
from utils.point_cloud_ops.correspondences import get_correspondences


def test_get_correspondences_basic():
    """Test basic correspondence finding functionality."""
    # Create simple test data
    src_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    tgt_points = torch.tensor(
        [
            [0.1, 0.0, 0.0],  # Close to src[0]
            [1.1, 0.0, 0.0],  # Close to src[1]
            [5.0, 0.0, 0.0],  # Far from all src points
        ],
        dtype=torch.float32,
    )

    radius = 0.5

    # Test without transform
    correspondences = get_correspondences(src_points, tgt_points, None, radius)

    # Should be [K, 2] format with [src_idx, tgt_idx] pairs
    assert correspondences.ndim == 2, "Correspondences should be 2D"
    assert correspondences.shape[1] == 2, "Each correspondence should have 2 indices"

    # Should find 2 correspondences (src[0]↔tgt[0], src[1]↔tgt[1])
    assert (
        correspondences.shape[0] == 2
    ), f"Expected 2 correspondences, got {correspondences.shape[0]}"

    # Check the actual correspondences
    corr_set = set((c[0].item(), c[1].item()) for c in correspondences)
    expected = {(0, 0), (1, 1)}  # src[0]↔tgt[0], src[1]↔tgt[1]
    assert corr_set == expected, f"Expected {expected}, got {corr_set}"


def test_get_correspondences_with_transform():
    """Test correspondence finding with transformation."""
    src_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    tgt_points = torch.tensor(
        [
            [1.1, 0.0, 0.0],  # Will match src[0] after translation
            [2.1, 0.0, 0.0],  # Will match src[1] after translation
        ],
        dtype=torch.float32,
    )

    # Translation transform: move src points by [1, 0, 0]
    transform = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    radius = 0.2

    correspondences = get_correspondences(src_points, tgt_points, transform, radius)

    # Should find 2 correspondences after transformation
    assert (
        correspondences.shape[0] == 2
    ), f"Expected 2 correspondences, got {correspondences.shape[0]}"
    assert correspondences.shape[1] == 2, "Each correspondence should have 2 indices"


def test_get_correspondences_with_dict_source():
    """Test correspondence finding with source as point cloud dictionary."""
    # Create source point cloud as dictionary
    src_dict = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    }

    tgt_points = torch.tensor(
        [
            [0.1, 0.0, 0.0],  # Close to src[0]
            [0.9, 0.0, 0.0],  # Close to src[1]
            [10.0, 0.0, 0.0],  # Far from all src points
        ],
        dtype=torch.float32,
    )

    radius = 0.3

    # Test with dict source and tensor target
    correspondences = get_correspondences(src_dict, tgt_points, None, radius)

    assert correspondences.ndim == 2, "Correspondences should be 2D"
    assert correspondences.shape[1] == 2, "Each correspondence should have 2 indices"
    assert (
        correspondences.shape[0] == 2
    ), f"Expected 2 correspondences, got {correspondences.shape[0]}"

    # Check correspondence indices
    corr_set = set((c[0].item(), c[1].item()) for c in correspondences)
    expected = {(0, 0), (1, 1)}  # src[0]↔tgt[0], src[1]↔tgt[1]
    assert corr_set == expected, f"Expected {expected}, got {corr_set}"


def test_get_correspondences_with_dict_target():
    """Test correspondence finding with target as point cloud dictionary."""
    src_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # Create target point cloud as dictionary
    tgt_dict = {
        'pos': torch.tensor(
            [
                [0.05, 0.0, 0.0],  # Close to src[0]
                [1.05, 0.0, 0.0],  # Close to src[1]
                [3.0, 0.0, 0.0],  # Far from all src points
            ],
            dtype=torch.float32,
        )
    }

    radius = 0.1

    # Test with tensor source and dict target
    correspondences = get_correspondences(src_points, tgt_dict, None, radius)

    assert correspondences.ndim == 2, "Correspondences should be 2D"
    assert correspondences.shape[1] == 2, "Each correspondence should have 2 indices"
    assert (
        correspondences.shape[0] == 2
    ), f"Expected 2 correspondences, got {correspondences.shape[0]}"

    # Check correspondence indices
    corr_set = set((c[0].item(), c[1].item()) for c in correspondences)
    expected = {(0, 0), (1, 1)}  # src[0]↔tgt[0], src[1]↔tgt[1]
    assert corr_set == expected, f"Expected {expected}, got {corr_set}"


def test_get_correspondences_with_both_dict():
    """Test correspondence finding with both source and target as dictionaries."""
    # Create both as dictionaries
    src_dict = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
        )
    }

    tgt_dict = {
        'pos': torch.tensor(
            [
                [0.0, 0.1, 0.0],  # Close to src[0]
                [1.0, 0.9, 0.0],  # Close to src[1]
                [5.0, 5.0, 0.0],  # Far from all src points
            ],
            dtype=torch.float32,
        )
    }

    radius = 0.25

    # Test with both as dictionaries
    correspondences = get_correspondences(src_dict, tgt_dict, None, radius)

    assert correspondences.ndim == 2, "Correspondences should be 2D"
    assert correspondences.shape[1] == 2, "Each correspondence should have 2 indices"
    assert (
        correspondences.shape[0] == 2
    ), f"Expected 2 correspondences, got {correspondences.shape[0]}"

    # Check correspondence indices
    corr_set = set((c[0].item(), c[1].item()) for c in correspondences)
    expected = {(0, 0), (1, 1)}  # src[0]↔tgt[0], src[1]↔tgt[1]
    assert corr_set == expected, f"Expected {expected}, got {corr_set}"


def test_get_correspondences_dict_with_transform():
    """Test correspondence finding with dictionary input and transformation."""
    src_dict = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    }

    tgt_dict = {
        'pos': torch.tensor(
            [
                [0.0, 0.05, 0.0],  # Will match src[0] after rotation (0,0,0) -> (0,0,0)
                [0.0, 1.05, 0.0],  # Will match src[1] after rotation (1,0,0) -> (0,1,0)
            ],
            dtype=torch.float32,
        )
    }

    # Rotation transform: 90 degrees around z-axis
    # (x,y,z) -> (-y,x,z), so (0,0,0)->(0,0,0) and (1,0,0)->(0,1,0)
    transform = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    radius = 0.1

    correspondences = get_correspondences(src_dict, tgt_dict, transform, radius)

    # Should find 2 correspondences after rotation
    assert (
        correspondences.shape[0] == 2
    ), f"Expected 2 correspondences, got {correspondences.shape[0]}"
    assert correspondences.shape[1] == 2, "Each correspondence should have 2 indices"

    # Check correspondence indices
    corr_set = set((c[0].item(), c[1].item()) for c in correspondences)
    expected = {(0, 0), (1, 1)}  # src[0]↔tgt[0], src[1]↔tgt[1] after transform
    assert corr_set == expected, f"Expected {expected}, got {corr_set}"
