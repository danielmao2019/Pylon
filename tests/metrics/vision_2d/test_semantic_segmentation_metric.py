import pytest
from typing import Dict
from metrics.vision_2d.semantic_segmentation_metric import SemanticSegmentationMetric
from runners.model_comparison import compare_scores, get_metric_directions
import torch


def create_datapoint(y_pred, y_true, idx=0):
    """Helper function to create datapoint for SemanticSegmentationMetric tests."""
    return {
        'inputs': {},  # Empty for these tests
        'outputs': y_pred,
        'labels': y_true,
        'meta_info': {'idx': idx}
    }


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
    datapoint = create_datapoint(y_pred, y_true)
    score: Dict[str, torch.Tensor] = metric(datapoint)
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
    for idx, (y_pred, y_true) in enumerate(zip(y_preds, y_trues, strict=True)):
        datapoint = create_datapoint(y_pred, y_true, idx)
        metric(datapoint)

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
        assert isinstance(result['per_datapoint'][key], list), f"{key=}, {result['per_datapoint'][key]=}"
        assert len(result['per_datapoint'][key]) == len(y_preds)


def test_semantic_segmentation_metric_directions():
    """Test that SemanticSegmentationMetric DIRECTIONS work with model comparison."""
    # Create metric instance
    metric = SemanticSegmentationMetric(num_classes=2)

    # Get directions
    directions = get_metric_directions(metric)

    # Verify expected directions are present
    expected_directions = {
        'mean_IoU': 1,
        'accuracy': 1,
        'mean_precision': 1,
        'mean_recall': 1,
        'mean_f1': 1,
    }

    assert directions == expected_directions, f"Expected {expected_directions}, got {directions}"


def test_compare_scores_with_semantic_segmentation_output():
    """Test score comparison using realistic semantic segmentation metric outputs."""
    # Create metric to get directions
    metric = SemanticSegmentationMetric(num_classes=2)
    directions = get_metric_directions(metric)

    # Simulate epoch 0 scores (poor performance, similar to real logs)
    epoch_0_scores = {
        'class_IoU': [0.9614, 0.0],  # Poor performance on class 1
        'mean_IoU': 0.4807,
        'class_tp': [3087418, 0],
        'class_tn': [0, 3087418],
        'class_fp': [123846, 0],
        'class_fn': [0, 123846],
        'class_accuracy': [0.9614, 0.9614],
        'class_precision': [0.9614, float('nan')],
        'class_recall': [1.0, 0.0],
        'class_f1': [0.9803, 0.0],
        'accuracy': 0.9614,
        'mean_precision': float('nan'),
        'mean_recall': 0.5,
        'mean_f1': 0.4902
    }

    # Simulate epoch 99 scores (good performance, similar to real logs)
    epoch_99_scores = {
        'class_IoU': [0.9819, 0.6530],  # Much better performance on class 1
        'mean_IoU': 0.8175,
        'class_tp': [3034996, 128363],
        'class_tn': [128363, 3034996],
        'class_fp': [24069, 23836],
        'class_fn': [23836, 24069],
        'class_accuracy': [0.9851, 0.9851],
        'class_precision': [0.9921, 0.8434],
        'class_recall': [0.9922, 0.8421],
        'class_f1': [0.9922, 0.8427],
        'accuracy': 0.9851,
        'mean_precision': 0.9178,
        'mean_recall': 0.9172,
        'mean_f1': 0.9175
    }

    # Test comparison: epoch 99 should be better than epoch 0
    result = compare_scores(
        current_scores=epoch_99_scores,
        best_scores=epoch_0_scores,
        order_config=False,  # Vector comparison
        metric_directions=directions
    )

    assert result is True, "Epoch 99 should be better than epoch 0"

    # Test reverse: epoch 0 should NOT be better than epoch 99
    result_reverse = compare_scores(
        current_scores=epoch_0_scores,
        best_scores=epoch_99_scores,
        order_config=False,
        metric_directions=directions
    )

    assert result_reverse is False, "Epoch 0 should not be better than epoch 99"


def test_comparison_ignores_metrics_without_directions():
    """Test that comparison only uses metrics with defined directions."""
    # Create metric to get directions
    metric = SemanticSegmentationMetric(num_classes=2)
    directions = get_metric_directions(metric)

    # Create scores that include metrics not in DIRECTIONS
    scores_with_extra_metrics = {
        'mean_IoU': 0.5,
        'accuracy': 0.9,
        'class_IoU': [0.8, 0.3],  # Not in DIRECTIONS - should be ignored
        'class_precision': [0.9, 0.7],  # Not in DIRECTIONS - should be ignored
        'some_other_metric': 42.0,  # Not in DIRECTIONS - should be ignored
    }

    baseline_scores = {
        'mean_IoU': 0.4,
        'accuracy': 0.8,
        'class_IoU': [0.9, 0.4],  # Better than current, but should be ignored
        'class_precision': [0.95, 0.8],  # Better than current, but should be ignored
        'some_other_metric': 100.0,  # Much better, but should be ignored
    }

    # Should return True because mean_IoU and accuracy are both better
    # (ignoring class_IoU, class_precision, and some_other_metric)
    result = compare_scores(
        current_scores=scores_with_extra_metrics,
        best_scores=baseline_scores,
        order_config=False,
        metric_directions=directions
    )

    assert result is True, "Should be better based on mean_IoU and accuracy only"


def test_nan_handling_in_scores():
    """Test that NaN values in scores don't break comparison."""
    metric = SemanticSegmentationMetric(num_classes=2)
    directions = get_metric_directions(metric)

    # Scores with NaN values (common in semantic segmentation)
    scores_with_nan = {
        'mean_IoU': 0.5,
        'accuracy': 0.9,
        'mean_precision': float('nan'),  # Common when no predictions for some classes
        'mean_recall': 0.7,
        'mean_f1': 0.6,
    }

    baseline_scores = {
        'mean_IoU': 0.4,
        'accuracy': 0.8,
        'mean_precision': 0.8,  # Better, but current is NaN
        'mean_recall': 0.6,
        'mean_f1': 0.5,
    }

    # Should not crash and should handle NaN appropriately
    result = compare_scores(
        current_scores=scores_with_nan,
        best_scores=baseline_scores,
        order_config=False,
        metric_directions=directions
    )

    # The result depends on NaN handling in the comparison logic
    # Main goal is that it doesn't crash
    assert result in [True, False, None], "Should return valid comparison result"
