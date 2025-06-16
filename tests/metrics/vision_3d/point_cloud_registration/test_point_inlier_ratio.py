import torch
import pytest
from metrics.vision_3d.point_cloud_registration.point_inlier_ratio import PointInlierRatio


@pytest.fixture
def metric():
    return PointInlierRatio()


def test_point_inlier_ratio_initialization():
    metric = PointInlierRatio()
    assert metric.DIRECTION == 1  # Higher is better


def test_point_inlier_ratio_perfect_match(metric):
    # Create point clouds and perfect correspondences
    src_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    tgt_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)

    # Perfect correspondences
    pred_correspondences = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)
    gt_correspondences = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)

    y_pred = {
        'src_points': src_points,
        'tgt_points': tgt_points,
        'correspondences': pred_correspondences
    }
    y_true = {'correspondences': gt_correspondences}

    result = metric(y_pred, y_true)
    assert 'point_inlier_ratio' in result
    assert torch.isclose(result['point_inlier_ratio'], torch.tensor(1.0))


def test_point_inlier_ratio_partial_match(metric):
    # Create point clouds with some correct and some incorrect correspondences
    src_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    tgt_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)

    # Some correct, some incorrect correspondences
    pred_correspondences = torch.tensor([[0, 0], [1, 2], [2, 1]], dtype=torch.long)
    gt_correspondences = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)

    y_pred = {
        'src_points': src_points,
        'tgt_points': tgt_points,
        'correspondences': pred_correspondences
    }
    y_true = {'correspondences': gt_correspondences}

    result = metric(y_pred, y_true)
    assert 'point_inlier_ratio' in result
    # Only the first correspondence is correct
    assert torch.isclose(result['point_inlier_ratio'], torch.tensor(1/3))


def test_point_inlier_ratio_no_match(metric):
    # Create point clouds with no matching correspondences
    src_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    tgt_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)

    # All incorrect correspondences
    pred_correspondences = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
    gt_correspondences = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)

    y_pred = {
        'src_points': src_points,
        'tgt_points': tgt_points,
        'correspondences': pred_correspondences
    }
    y_true = {'correspondences': gt_correspondences}

    result = metric(y_pred, y_true)
    assert 'point_inlier_ratio' in result
    assert torch.isclose(result['point_inlier_ratio'], torch.tensor(0.0))


def test_point_inlier_ratio_invalid_inputs(metric):
    # Test with invalid input shapes
    with pytest.raises(AssertionError):
        src_points = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # Invalid shape
        tgt_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        correspondences = torch.tensor([[0, 0]], dtype=torch.long)

        y_pred = {
            'src_points': src_points,
            'tgt_points': tgt_points,
            'correspondences': correspondences
        }
        y_true = {'correspondences': correspondences}
        metric(y_pred, y_true)

    # Test with invalid correspondence shape
    with pytest.raises(AssertionError):
        src_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        tgt_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        correspondences = torch.tensor([[0, 0, 0]], dtype=torch.long)  # Invalid shape

        y_pred = {
            'src_points': src_points,
            'tgt_points': tgt_points,
            'correspondences': correspondences
        }
        y_true = {'correspondences': correspondences}
        metric(y_pred, y_true)

    # Test with missing required keys
    with pytest.raises(AssertionError):
        src_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        tgt_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        correspondences = torch.tensor([[0, 0]], dtype=torch.long)

        y_pred = {
            'src_points': src_points,
            'tgt_points': tgt_points,
            # Missing 'correspondences' key
        }
        y_true = {'correspondences': correspondences}
        metric(y_pred, y_true)
