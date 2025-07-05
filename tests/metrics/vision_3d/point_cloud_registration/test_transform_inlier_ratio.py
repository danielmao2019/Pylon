import torch
import pytest
from metrics.vision_3d.point_cloud_registration.transform_inlier_ratio import TransformInlierRatio


def create_datapoint(src_pc, tgt_pc, transform, idx=0):
    """Helper function to create datapoint for TransformInlierRatio tests.

    Args:
        src_pc: Source point cloud
        tgt_pc: Target point cloud
        transform: Predicted transformation matrix
    """
    return {
        'inputs': {
            'src_pc': src_pc,
            'tgt_pc': tgt_pc
        },
        'outputs': {'transform': transform},
        'labels': {},  # Not used by this metric
        'meta_info': {'idx': idx}
    }


@pytest.fixture
def metric():
    return TransformInlierRatio(threshold=0.1)


def test_transform_inlier_ratio_initialization():
    metric = TransformInlierRatio(threshold=0.1)
    assert metric.threshold == 0.1
    assert metric.DIRECTIONS == {'inlier_ratio': 1}  # Higher is better


def test_transform_inlier_ratio_perfect_match(metric):
    """Test with identity transform - all points should be inliers."""
    # Create source and target point clouds that are identical
    src_pc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    tgt_pc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    transform = torch.eye(4, dtype=torch.float32)  # Identity transform

    datapoint = create_datapoint(src_pc, tgt_pc, transform)
    result = metric(datapoint)

    assert 'inlier_ratio' in result
    assert torch.isclose(result['inlier_ratio'], torch.tensor(1.0))


def test_transform_inlier_ratio_partial_match(metric):
    """Test with some inliers and some outliers."""
    # Source points
    src_pc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)

    # Target points: first two close to source after transform, third far away
    tgt_pc = torch.tensor([[0.05, 0.05, 0.05], [1.05, 1.05, 1.05], [10.0, 10.0, 10.0]], dtype=torch.float32)

    transform = torch.eye(4, dtype=torch.float32)  # Identity transform

    datapoint = create_datapoint(src_pc, tgt_pc, transform)
    result = metric(datapoint)

    assert 'inlier_ratio' in result
    # First two points should be inliers (distance ~0.087 < threshold 0.1)
    # Third point should be outlier (distance ~13.86 > threshold 0.1)
    assert torch.isclose(result['inlier_ratio'], torch.tensor(2/3), atol=1e-3)


def test_transform_inlier_ratio_with_translation(metric):
    """Test with translation transform."""
    # Source points
    src_pc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)

    # Target points: source points translated by [1, 1, 1]
    tgt_pc = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.float32)

    # Translation transform by [1, 1, 1]
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, 3] = torch.tensor([1.0, 1.0, 1.0])

    datapoint = create_datapoint(src_pc, tgt_pc, transform)
    result = metric(datapoint)

    assert 'inlier_ratio' in result
    assert torch.isclose(result['inlier_ratio'], torch.tensor(1.0))


def test_transform_inlier_ratio_batch_transform(metric):
    """Test with batched transform input (1, 4, 4)."""
    src_pc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    tgt_pc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    transform = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # (1, 4, 4)

    datapoint = create_datapoint(src_pc, tgt_pc, transform)
    result = metric(datapoint)

    assert 'inlier_ratio' in result
    assert torch.isclose(result['inlier_ratio'], torch.tensor(1.0))


def test_transform_inlier_ratio_no_inliers(metric):
    """Test case where no points are inliers."""
    # Source points
    src_pc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)

    # Target points very far from source
    tgt_pc = torch.tensor([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], dtype=torch.float32)

    transform = torch.eye(4, dtype=torch.float32)  # Identity transform

    datapoint = create_datapoint(src_pc, tgt_pc, transform)
    result = metric(datapoint)

    assert 'inlier_ratio' in result
    assert torch.isclose(result['inlier_ratio'], torch.tensor(0.0))


def test_transform_inlier_ratio_invalid_inputs(metric):
    """Test with invalid input shapes and types."""
    # Test with invalid source point dimensions (should be 3D)
    with pytest.raises(AssertionError):
        src_pc = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # Only 2D
        tgt_pc = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # 3D
        transform = torch.eye(4, dtype=torch.float32)

        datapoint = create_datapoint(src_pc, tgt_pc, transform)
        metric(datapoint)

    # Test with invalid transform shape
    with pytest.raises(AssertionError):
        src_pc = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        tgt_pc = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        transform = torch.eye(3, dtype=torch.float32)  # Wrong size (should be 4x4)

        datapoint = create_datapoint(src_pc, tgt_pc, transform)
        metric(datapoint)

    # Test with missing required keys
    with pytest.raises(AssertionError):
        datapoint = {
            'inputs': {'src_pc': torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)},  # Missing tgt_pc
            'outputs': {'transform': torch.eye(4, dtype=torch.float32)},
            'labels': {},
            'meta_info': {'idx': 0}
        }
        metric(datapoint)


def test_transform_inlier_ratio_different_point_counts():
    """Test with different numbers of source and target points."""
    metric = TransformInlierRatio(threshold=0.1)

    # 3 source points, 5 target points
    src_pc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    tgt_pc = torch.tensor([
        [0.05, 0.05, 0.05],   # Close to src[0]
        [1.05, 1.05, 1.05],   # Close to src[1]
        [2.05, 2.05, 2.05],   # Close to src[2]
        [10.0, 10.0, 10.0],   # Far from all
        [20.0, 20.0, 20.0]    # Far from all
    ], dtype=torch.float32)

    transform = torch.eye(4, dtype=torch.float32)

    datapoint = create_datapoint(src_pc, tgt_pc, transform)
    result = metric(datapoint)

    assert 'inlier_ratio' in result
    # All 3 source points should find close matches in target
    assert torch.isclose(result['inlier_ratio'], torch.tensor(1.0), atol=1e-3)


def test_transform_inlier_ratio_threshold_sensitivity():
    """Test sensitivity to threshold parameter."""
    # Create points with specific distances
    src_pc = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    tgt_pc = torch.tensor([[0.05, 0.0, 0.0]], dtype=torch.float32)  # Distance 0.05
    transform = torch.eye(4, dtype=torch.float32)

    datapoint = create_datapoint(src_pc, tgt_pc, transform)

    # With threshold 0.1, should be inlier
    metric_loose = TransformInlierRatio(threshold=0.1)
    result_loose = metric_loose(datapoint)
    assert torch.isclose(result_loose['inlier_ratio'], torch.tensor(1.0))

    # With threshold 0.01, should be outlier
    metric_strict = TransformInlierRatio(threshold=0.01)
    result_strict = metric_strict(datapoint)
    assert torch.isclose(result_strict['inlier_ratio'], torch.tensor(0.0))
