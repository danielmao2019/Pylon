import torch
import pytest
from metrics.vision_3d.point_cloud_registration.inlier_ratio import InlierRatio


def create_datapoint(src_points, tgt_points, gt_transform, idx=0):
    """Helper function to create datapoint for InlierRatio tests.

    Args:
        src_points: Predicted source points (correspondences)
        tgt_points: Predicted target points (correspondences)
        gt_transform: Ground truth transformation matrix
    """
    return {
        'inputs': {},  # Not used by InlierRatio
        'outputs': {
            'src_pc': src_points,  # Predicted source correspondences
            'tgt_pc': tgt_points   # Predicted target correspondences
        },
        'labels': {'transform': gt_transform},  # Ground truth transform
        'meta_info': {'idx': idx}
    }


@pytest.fixture
def metric():
    return InlierRatio(threshold=0.1)


def test_inlier_ratio_initialization():
    metric = InlierRatio(threshold=0.1)
    assert metric.threshold == 0.1
    assert metric.DIRECTION == 1  # Higher is better


def test_inlier_ratio_perfect_match(metric):
    # Create perfect correspondences: when GT transform is applied to src, they exactly match tgt
    src_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    tgt_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    gt_transform = torch.eye(4).unsqueeze(0)  # Identity transform

    datapoint = create_datapoint(src_points, tgt_points, gt_transform)
    result = metric(datapoint)
    assert 'inlier_ratio' in result
    assert torch.isclose(result['inlier_ratio'], torch.tensor(1.0))


def test_inlier_ratio_partial_match(metric):
    # Create correspondences with some good and some bad matches
    # After applying GT identity transform, src points should be:
    # [0,0,0] -> [0,0,0] (distance 0.0, within threshold 0.1) ✓
    # [1,1,1] -> [1.05,1.05,1.05] (distance ~0.087, within threshold 0.1) ✓
    # [2,2,2] -> [3,3,3] (distance ~1.73, beyond threshold 0.1) ✗
    src_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    tgt_points = torch.tensor([[0.0, 0.0, 0.0], [1.05, 1.05, 1.05], [3.0, 3.0, 3.0]], dtype=torch.float32)
    gt_transform = torch.eye(4).unsqueeze(0)  # Identity transform

    datapoint = create_datapoint(src_points, tgt_points, gt_transform)
    result = metric(datapoint)
    assert 'inlier_ratio' in result
    # Two out of three correspondences should be inliers
    assert torch.isclose(result['inlier_ratio'], torch.tensor(2/3))


def test_inlier_ratio_with_transform(metric):
    # Test with a non-identity GT transform
    # If GT transform translates by [1,1,1], then:
    # src [0,0,0] + [1,1,1] = [1,1,1] should match tgt [1,1,1] ✓
    # src [1,1,1] + [1,1,1] = [2,2,2] should match tgt [2,2,2] ✓
    # src [2,2,2] + [1,1,1] = [3,3,3] should match tgt [3,3,3] ✓
    src_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    tgt_points = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.float32)

    # Translation transform by [1,1,1]
    gt_transform = torch.eye(4).unsqueeze(0)
    gt_transform[0, :3, 3] = torch.tensor([1.0, 1.0, 1.0])

    datapoint = create_datapoint(src_points, tgt_points, gt_transform)
    result = metric(datapoint)
    assert 'inlier_ratio' in result
    assert torch.isclose(result['inlier_ratio'], torch.tensor(1.0))


def test_inlier_ratio_invalid_inputs(metric):
    # Test with invalid point dimensions (should be 3D)
    with pytest.raises(AssertionError):
        src_points = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # Only 2D
        tgt_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # 3D
        gt_transform = torch.eye(4).unsqueeze(0)

        datapoint = create_datapoint(src_points, tgt_points, gt_transform)
        metric(datapoint)

    # Test with mismatched correspondence counts
    with pytest.raises(AssertionError):
        src_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)  # 2 points
        tgt_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # 1 point
        gt_transform = torch.eye(4).unsqueeze(0)

        datapoint = create_datapoint(src_points, tgt_points, gt_transform)
        metric(datapoint)

    # Test with invalid transform shape
    with pytest.raises(AssertionError):
        src_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        tgt_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        gt_transform = torch.eye(3)  # Wrong size (should be 4x4)

        datapoint = create_datapoint(src_points, tgt_points, gt_transform)
        metric(datapoint)
