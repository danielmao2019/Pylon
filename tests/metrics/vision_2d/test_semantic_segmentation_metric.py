import pytest
from typing import Dict
from metrics.vision_2d.semantic_segmentation_metric import SemanticSegmentationMetric
import torch


@pytest.mark.parametrize("y_pred, y_true, expected_output", [
    (
        torch.tensor([[
            [[1., 1., 5.], [2., 0., 7.], [8., 1., 7.]],
            [[1., 4., 4.], [3., 2., 4.], [7., 4., 8.]],
            [[4., 2., 3.], [2., 9., 1.], [0., 8., 7.]]]], dtype=torch.float32).permute((0, 3, 1, 2)),
        torch.tensor([[
            [2, 2, 2],
            [1, 2, 0],
            [2, 0, 2]]], dtype=torch.int64),
        {
            'IoU': torch.tensor([0/4, 1/3, 3/7], dtype=torch.float32),
            'tp': torch.tensor([0, 1, 3], dtype=torch.int64),
            'tn': torch.tensor([5, 6, 2], dtype=torch.int64),
            'fp': torch.tensor([2, 2, 1], dtype=torch.int64),
            'fn': torch.tensor([2, 0, 3], dtype=torch.int64),
        },
    ),
])
def test_semantic_segmentation_iou(y_pred, y_true, expected_output):
    """Tests IoU computation for multiple classes."""
    metric = SemanticSegmentationMetric(num_classes=len(expected_output['IoU']))
    score: Dict[str, torch.Tensor] = metric(y_pred, y_true)
    # check IoU computation
    iou = score['IoU']
    expected_iou = expected_output['IoU']
    if torch.isnan(expected_iou).any():
        for i, val in enumerate(expected_iou):
            if torch.isnan(val):
                assert torch.isnan(iou[i])  # Ensure NaN for missing class
            else:
                torch.testing.assert_close(iou[i], val)
    else:
        torch.testing.assert_close(iou, expected_iou)
    # check confusion matrix
    for name in ['tp', 'tn', 'fp', 'fn']:
        assert torch.equal(score[name], expected_output[name])
