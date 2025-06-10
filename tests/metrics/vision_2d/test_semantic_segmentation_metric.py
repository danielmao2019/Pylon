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
            'class_IoU': torch.tensor([0/4, 1/3, 3/7], dtype=torch.float32),
            'class_tp': torch.tensor([0, 1, 3], dtype=torch.int64),
            'class_tn': torch.tensor([5, 6, 2], dtype=torch.int64),
            'class_fp': torch.tensor([2, 2, 1], dtype=torch.int64),
            'class_fn': torch.tensor([2, 0, 3], dtype=torch.int64),
        },
    ),
])
def test_semantic_segmentation_metric_call(y_pred, y_true, expected_output):
    """Tests IoU computation for multiple classes."""
    metric = SemanticSegmentationMetric(num_classes=len(expected_output['class_IoU']))
    score: Dict[str, torch.Tensor] = metric(y_pred, y_true)
    # check IoU computation
    iou = score['class_IoU']
    expected_iou = expected_output['class_IoU']
    if torch.isnan(expected_iou).any():
        for i, val in enumerate(expected_iou):
            if torch.isnan(val):
                assert torch.isnan(iou[i])  # Ensure NaN for missing class
            else:
                torch.testing.assert_close(iou[i], val)
    else:
        torch.testing.assert_close(iou, expected_iou)
    # check confusion matrix
    for name in ['class_tp', 'class_tn', 'class_fp', 'class_fn']:
        assert torch.equal(score[name], expected_output[name])


@pytest.mark.parametrize("y_preds, y_trues", [
    (
        [
            torch.tensor([[
                [[1., 1., 5.], [2., 0., 7.], [8., 1., 7.]],
                [[1., 4., 4.], [3., 2., 4.], [7., 4., 8.]],
                [[4., 2., 3.], [2., 9., 1.], [0., 8., 7.]]]], dtype=torch.float32).permute((0, 3, 1, 2)),
            torch.tensor([[
                [[1., 1., 5.], [2., 0., 7.], [8., 1., 7.]],
                [[1., 4., 4.], [3., 2., 4.], [7., 4., 8.]],
                [[4., 2., 3.], [2., 9., 1.], [0., 8., 7.]]]], dtype=torch.float32).permute((0, 3, 1, 2)),
        ],
        [
            torch.tensor([[
                [2, 2, 2],
                [1, 2, 0],
                [2, 0, 2]]], dtype=torch.int64),
            torch.tensor([[
                [2, 2, 2],
                [1, 2, 0],
                [2, 0, 2]]], dtype=torch.int64),
        ],
    ),
])
def test_semantic_segmentation_metric_summarize(y_preds, y_trues):
    """Tests semantic segmentation metric summarization across multiple datapoints."""
    metric = SemanticSegmentationMetric(num_classes=3)

    # Compute scores for each datapoint
    for y_pred, y_true in zip(y_preds, y_trues):
        metric(y_pred, y_true)

    # Summarize results
    result = metric.summarize()

    # Check structure
    assert result.keys() == {'aggregated', 'per_datapoint'}
    expected_keys = {
        'class_IoU', 'mean_IoU',
        'class_tp', 'class_tn', 'class_fp', 'class_fn',
        'class_accuracy', 'class_precision', 'class_recall', 'class_f1',
        'accuracy', 'mean_precision', 'mean_recall', 'mean_f1',
    }
    assert result['aggregated'].keys() == expected_keys
    for key in expected_keys:
        assert isinstance(result['aggregated'][key], torch.Tensor)
    assert result['per_datapoint'].keys() == expected_keys
    for key in expected_keys:
        assert isinstance(result['per_datapoint'][key], list)
        assert len(result['per_datapoint'][key]) == len(y_preds)
