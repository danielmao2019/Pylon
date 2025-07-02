from typing import Dict
import pytest
import torch
from runners.model_comparison import get_metric_directions, compare_scores, reduce_scores_to_scalar
from metrics.wrappers.single_task_metric import SingleTaskMetric
from metrics.wrappers.multi_task_metric import MultiTaskMetric
from metrics.wrappers.hybrid_metric import HybridMetric
from metrics.vision_2d.semantic_segmentation_metric import SemanticSegmentationMetric


class MockMetricWithDirection(SingleTaskMetric):
    """Mock metric with DIRECTIONS attribute."""
    DIRECTIONS = {"accuracy": 1}
    
    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"accuracy": torch.tensor(0.95)}


class MockMetricNegativeDirection(SingleTaskMetric):
    """Mock metric with negative DIRECTIONS."""
    DIRECTIONS = {"loss": -1}
    
    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"loss": torch.tensor(0.1)}


class MockNoDirectionMetric:
    """Mock metric with no DIRECTION attribute anywhere."""
    
    def __init__(self):
        self.value = 42


class MockChangeStarLikeMetric(SingleTaskMetric):
    """Mock metric similar to ChangeStarMetric structure."""
    
    # Define explicit directions for what this metric actually produces
    DIRECTIONS = {
        'mean_IoU': 1, 'accuracy': 1, 'mean_precision': 1, 
        'mean_recall': 1, 'mean_f1': 1
    }
    
    def __init__(self):
        super().__init__()
        self.change_metric = SemanticSegmentationMetric(num_classes=2, use_buffer=False)
        self.semantic_metric = SemanticSegmentationMetric(num_classes=5, use_buffer=False)
    
    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"combined_score": torch.tensor(0.8)}


def test_get_metric_directions_single_metric():
    """Test get_metric_directions with single metric having DIRECTIONS."""
    metric = MockMetricWithDirection()
    directions = get_metric_directions(metric)
    assert directions == {"accuracy": 1}


def test_get_metric_directions_negative_direction():
    """Test get_metric_directions with negative DIRECTIONS."""
    metric = MockMetricNegativeDirection()
    directions = get_metric_directions(metric)
    assert directions == {"loss": -1}


def test_get_metric_directions_nested_metrics():
    """Test get_metric_directions with nested metrics (ChangeStarMetric-like structure)."""
    metric = MockChangeStarLikeMetric()
    directions = get_metric_directions(metric)
    
    # Should use the MockChangeStarLikeMetric's own DIRECTIONS
    expected = {
        'mean_IoU': 1, 'accuracy': 1, 'mean_precision': 1, 
        'mean_recall': 1, 'mean_f1': 1
    }
    assert directions == expected


def test_get_metric_directions_multi_task_metric():
    """Test get_metric_directions with MultiTaskMetric."""
    task_configs = {
        'classification': {
            'class': MockMetricWithDirection,
            'args': {}
        },
        'regression': {
            'class': MockMetricNegativeDirection,
            'args': {}
        }
    }
    metric = MultiTaskMetric(task_configs)
    directions = get_metric_directions(metric)
    expected = {
        "classification": {"accuracy": 1},  # Full DIRECTIONS from MockMetricWithDirection
        "regression": {"loss": -1}          # Full DIRECTIONS from MockMetricNegativeDirection
    }
    assert directions == expected


def test_get_metric_directions_hybrid_metric():
    """Test get_metric_directions with HybridMetric."""
    metrics_cfg = [
        {
            'class': MockMetricWithDirection,
            'args': {}
        },
        {
            'class': MockMetricNegativeDirection,
            'args': {}
        }
    ]
    metric = HybridMetric(metrics_cfg=metrics_cfg)
    directions = get_metric_directions(metric)
    
    # HybridMetric merges all DIRECTIONS from component metrics
    expected = {
        "accuracy": 1,  # From MockMetricWithDirection.DIRECTIONS
        "loss": -1      # From MockMetricNegativeDirection.DIRECTIONS
    }
    assert directions == expected


def test_get_metric_directions_semantic_segmentation_metric():
    """Test get_metric_directions with actual SemanticSegmentationMetric."""
    metric = SemanticSegmentationMetric(num_classes=10, use_buffer=False)
    directions = get_metric_directions(metric)
    expected = {
        'mean_IoU': 1, 'accuracy': 1, 'mean_precision': 1, 
        'mean_recall': 1, 'mean_f1': 1
    }
    assert directions == expected


def test_get_metric_directions_no_direction_fails():
    """Test that get_metric_directions fails when no DIRECTIONS found."""
    metric = MockNoDirectionMetric()
    with pytest.raises(AttributeError, match="has no DIRECTIONS attribute"):
        get_metric_directions(metric)


def test_get_metric_directions_none_metric_fails():
    """Test that get_metric_directions fails when metric is None."""
    with pytest.raises(AssertionError, match="Metric cannot be None"):
        get_metric_directions(None)


def test_get_metric_directions_invalid_direction_value():
    """Test that get_metric_directions fails with invalid DIRECTIONS values."""
    class InvalidDirectionMetric:
        DIRECTIONS = {"score": 0}  # Invalid - should be 1 or -1
    
    metric = InvalidDirectionMetric()
    with pytest.raises(AssertionError, match="DIRECTION at 'score' must be -1 or 1"):
        get_metric_directions(metric)


@pytest.mark.parametrize("metric_directions,scores1,scores2,expected", [
    ({"accuracy": 1}, {"accuracy": 0.9}, {"accuracy": 0.8}, True),  # Higher better
    ({"loss": -1}, {"loss": 0.1}, {"loss": 0.2}, True),  # Lower better
    ({"accuracy": 1}, {"accuracy": 0.8}, {"accuracy": 0.9}, False),  # Higher better but worse
    ({"score": 1}, {"score": 0.9}, {"score": 0.8}, True),  # Single score - higher better
    ({"loss": -1}, {"loss": 0.1}, {"loss": 0.2}, True),  # Single score - lower better
])
def test_compare_scores_with_complex_directions(metric_directions, scores1, scores2, expected):
    """Test compare_scores with various direction configurations."""
    # Use True for order_config (equal weight average) 
    result = compare_scores(scores1, scores2, True, metric_directions)
    assert result == expected


def test_reduce_scores_to_scalar_multiple_metrics():
    """Test reduce_scores_to_scalar with multiple score keys."""
    scores = {
        "accuracy": torch.tensor(0.9),
        "f1_score": torch.tensor(0.85),
        "precision": torch.tensor(0.88)
    }
    order_config = {"accuracy": 0.5, "f1_score": 0.3, "precision": 0.2}
    metric_directions = {"accuracy": 1, "f1_score": 1, "precision": 1}
    
    result = reduce_scores_to_scalar(scores, order_config, metric_directions)
    expected = 0.9 * 0.5 + 0.85 * 0.3 + 0.88 * 0.2
    assert abs(result - expected) < 1e-6


def test_reduce_scores_to_scalar_equal_weights():
    """Test reduce_scores_to_scalar with equal weights (True config)."""
    scores = {
        "metric1": torch.tensor(0.8),
        "metric2": torch.tensor(0.9)
    }
    metric_directions = {"metric1": 1, "metric2": 1}
    
    result = reduce_scores_to_scalar(scores, True, metric_directions)
    expected = (0.8 + 0.9) / 2
    assert abs(result - expected) < 1e-6


def test_reduce_scores_to_scalar_with_explicit_directions():
    """Test reduce_scores_to_scalar with explicit directions for each score."""
    scores = {
        "accuracy": torch.tensor(0.9),
        "f1_score": torch.tensor(0.85)
    }
    # Explicit directions for each score key
    metric_directions = {"accuracy": 1, "f1_score": 1}
    
    result = reduce_scores_to_scalar(scores, True, metric_directions)
    expected = (0.9 + 0.85) / 2
    assert abs(result - expected) < 1e-6


def test_integration_complex_metric_with_early_stopping():
    """Integration test: complex metric directions work with early stopping logic."""
    # Create a complex metric similar to real usage
    task_configs = {
        'segmentation': {
            'class': SemanticSegmentationMetric,
            'args': {'num_classes': 10, 'use_buffer': False}
        }
    }
    metric = MultiTaskMetric(task_configs)
    
    # Test that directions can be extracted
    directions = get_metric_directions(metric)
    assert "segmentation" in directions
    assert isinstance(directions["segmentation"], dict)  # Should be nested dict
    assert "mean_IoU" in directions["segmentation"]     # Should contain semantic seg metrics
    
    # Test score comparison - use nested structure to match the directions
    scores1 = {"segmentation": {"mean_IoU": 0.9, "accuracy": 0.85}}
    scores2 = {"segmentation": {"mean_IoU": 0.85, "accuracy": 0.8}}
    
    # Test comparison (should indicate improvement since scores1 > scores2)
    result = compare_scores(scores1, scores2, True, directions)
    assert result == True
