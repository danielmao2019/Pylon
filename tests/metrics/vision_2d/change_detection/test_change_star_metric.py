import pytest
import torch
from metrics.vision_2d.change_detection.change_star_metric import ChangeStarMetric


@pytest.mark.parametrize("y_pred, y_true", [
    (
        {
            'change': torch.tensor([[
                [[0.9], [0.1]],
                [[0.2], [0.8]],
            ]], dtype=torch.float32),  # [1, 1, 2, 2]
            'semantic_1': torch.tensor([[
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.3, 0.7], [0.6, 0.4]],
            ]], dtype=torch.float32),  # [1, 2, 2, 2]
            'semantic_2': torch.tensor([[
                [[0.8, 0.2], [0.1, 0.9]],
                [[0.4, 0.6], [0.7, 0.3]],
            ]], dtype=torch.float32),  # [1, 2, 2, 2]
        },
        {
            'change': torch.tensor([[
                [1, 0],
                [0, 1],
            ]], dtype=torch.int64),  # [1, 2, 2]
            'semantic_1': torch.tensor([[
                [1, 0],
                [0, 1],
            ]], dtype=torch.int64),  # [1, 2, 2]
            'semantic_2': torch.tensor([[
                [1, 0],
                [0, 1],
            ]], dtype=torch.int64),  # [1, 2, 2]
        },
    ),
])
def test_change_star_metric_call(y_pred, y_true):
    """Tests change star metric computation for a single datapoint."""
    metric = ChangeStarMetric()
    score = metric(y_pred, y_true)
    
    # Check structure
    expected_categories = {'change', 'semantic_1', 'semantic_2'}
    assert set(score.keys()) == expected_categories
    
    # Check each category has the expected metrics
    expected_metrics = {'IoU', 'class_tp', 'class_tn', 'class_fp', 'class_fn'}
    for category in expected_categories:
        assert set(score[category].keys()) == expected_metrics
        for metric in expected_metrics:
            assert isinstance(score[category][metric], torch.Tensor)


@pytest.mark.parametrize("y_preds, y_trues", [
    (
        [
            {
                'change': torch.tensor([[
                    [[0.9], [0.1]],
                    [[0.2], [0.8]],
                ]], dtype=torch.float32),  # [1, 1, 2, 2]
                'semantic_1': torch.tensor([[
                    [[0.9, 0.1], [0.2, 0.8]],
                    [[0.3, 0.7], [0.6, 0.4]],
                ]], dtype=torch.float32),  # [1, 2, 2, 2]
                'semantic_2': torch.tensor([[
                    [[0.8, 0.2], [0.1, 0.9]],
                    [[0.4, 0.6], [0.7, 0.3]],
                ]], dtype=torch.float32),  # [1, 2, 2, 2]
            },
            {
                'change': torch.tensor([[
                    [[0.8], [0.2]],
                    [[0.3], [0.7]],
                ]], dtype=torch.float32),  # [1, 1, 2, 2]
                'semantic_1': torch.tensor([[
                    [[0.8, 0.2], [0.3, 0.7]],
                    [[0.4, 0.6], [0.5, 0.5]],
                ]], dtype=torch.float32),  # [1, 2, 2, 2]
                'semantic_2': torch.tensor([[
                    [[0.7, 0.3], [0.2, 0.8]],
                    [[0.5, 0.5], [0.6, 0.4]],
                ]], dtype=torch.float32),  # [1, 2, 2, 2]
            },
        ],
        [
            {
                'change': torch.tensor([[
                    [1, 0],
                    [0, 1],
                ]], dtype=torch.int64),  # [1, 2, 2]
                'semantic_1': torch.tensor([[
                    [1, 0],
                    [0, 1],
                ]], dtype=torch.int64),  # [1, 2, 2]
                'semantic_2': torch.tensor([[
                    [1, 0],
                    [0, 1],
                ]], dtype=torch.int64),  # [1, 2, 2]
            },
            {
                'change': torch.tensor([[
                    [1, 0],
                    [0, 1],
                ]], dtype=torch.int64),  # [1, 2, 2]
                'semantic_1': torch.tensor([[
                    [1, 0],
                    [0, 1],
                ]], dtype=torch.int64),  # [1, 2, 2]
                'semantic_2': torch.tensor([[
                    [1, 0],
                    [0, 1],
                ]], dtype=torch.int64),  # [1, 2, 2]
            },
        ],
    ),
])
def test_change_star_metric_summarize(y_preds, y_trues):
    """Tests change star metric summarization across multiple datapoints."""
    metric = ChangeStarMetric()
    
    # Compute scores for each datapoint
    for y_pred, y_true in zip(y_preds, y_trues):
        metric(y_pred, y_true)
    
    # Summarize results
    result = metric.summarize()
    
    # Check structure
    assert set(result.keys()) == {'aggregated', 'per_datapoint'}
    
    # Check aggregated structure
    expected_categories = {'change', 'semantic_1', 'semantic_2'}
    expected_aggregated_keys = {
        'class_IoU', 'mean_IoU',
        'class_tp', 'class_tn', 'class_fp', 'class_fn',
        'class_accuracy', 'class_precision', 'class_recall', 'class_f1',
        'accuracy', 'mean_precision', 'mean_recall', 'mean_f1',
    }
    for category in expected_categories:
        assert category in result['aggregated']
        assert set(result['aggregated'][category].keys()) == expected_aggregated_keys
    
    # Check per_datapoint structure
    expected_per_datapoint_keys = {
        'IoU',
        'class_tp', 'class_tn', 'class_fp', 'class_fn',
        'class_accuracy', 'class_precision', 'class_recall', 'class_f1',
        'accuracy', 'mean_precision', 'mean_recall', 'mean_f1',
    }
    for category in expected_categories:
        assert category in result['per_datapoint']
        assert set(result['per_datapoint'][category].keys()) == expected_per_datapoint_keys
