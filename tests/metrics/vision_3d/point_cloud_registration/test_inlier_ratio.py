import torch
import pytest
from metrics.vision_3d.point_cloud_registration.inlier_ratio import InlierRatio


@pytest.fixture
def metric():
    return InlierRatio(threshold=0.1)


def test_inlier_ratio_initialization():
    metric = InlierRatio(threshold=0.1)
    assert metric.threshold == 0.1
    assert metric.DIRECTION == 1  # Higher is better


def test_inlier_ratio_perfect_match(metric):
    # Create perfectly matching point clouds
    src_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    tgt_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    transform = torch.eye(4).unsqueeze(0)  # Identity transform

    y_pred = {'src_points': src_points, 'tgt_points': tgt_points}
    y_true = {'transform': transform}

    result = metric(y_pred, y_true)
    assert 'inlier_ratio' in result
    assert torch.isclose(result['inlier_ratio'], torch.tensor(1.0))


def test_inlier_ratio_partial_match(metric):
    # Create point clouds with some points matching and some not
    src_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    tgt_points = torch.tensor([[0.0, 0.0, 0.0], [1.1, 1.1, 1.1], [3.0, 3.0, 3.0]], dtype=torch.float32)
    transform = torch.eye(4).unsqueeze(0)  # Identity transform

    y_pred = {'src_points': src_points, 'tgt_points': tgt_points}
    y_true = {'transform': transform}

    result = metric(y_pred, y_true)
    assert 'inlier_ratio' in result
    # Only the first point should be an inlier
    assert torch.isclose(result['inlier_ratio'], torch.tensor(1/3))


def test_inlier_ratio_with_transform(metric):
    # Test with a non-identity transform
    src_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    tgt_points = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.float32)
    
    # Translation transform
    transform = torch.eye(4).unsqueeze(0)
    transform[0, :3, 3] = torch.tensor([1.0, 1.0, 1.0])

    y_pred = {'src_points': src_points, 'tgt_points': tgt_points}
    y_true = {'transform': transform}

    result = metric(y_pred, y_true)
    assert 'inlier_ratio' in result
    assert torch.isclose(result['inlier_ratio'], torch.tensor(1.0))


def test_inlier_ratio_invalid_inputs(metric):
    # Test with invalid input shapes
    with pytest.raises(AssertionError):
        src_points = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # Invalid shape
        tgt_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        transform = torch.eye(4).unsqueeze(0)
        
        y_pred = {'src_points': src_points, 'tgt_points': tgt_points}
        y_true = {'transform': transform}
        metric(y_pred, y_true)

    # Test with mismatched point cloud sizes
    with pytest.raises(AssertionError):
        src_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        tgt_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # Different size
        transform = torch.eye(4).unsqueeze(0)
        
        y_pred = {'src_points': src_points, 'tgt_points': tgt_points}
        y_true = {'transform': transform}
        metric(y_pred, y_true)

    # Test with invalid transform shape
    with pytest.raises(AssertionError):
        src_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        tgt_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        transform = torch.eye(4)  # Missing batch dimension
        
        y_pred = {'src_points': src_points, 'tgt_points': tgt_points}
        y_true = {'transform': transform}
        metric(y_pred, y_true)
