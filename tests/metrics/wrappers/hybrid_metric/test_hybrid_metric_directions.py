"""
Test HybridMetric DIRECTIONS overlap detection.
"""
import pytest
from metrics.wrappers.hybrid_metric import HybridMetric
from metrics.wrappers.single_task_metric import SingleTaskMetric


class MockMetricA(SingleTaskMetric):
    """Mock metric with specific DIRECTIONS."""
    DIRECTIONS = {"accuracy": 1, "precision": 1}

    def _compute_score(self, y_pred, y_true):
        return {"accuracy": 0.9, "precision": 0.85}


class MockMetricB(SingleTaskMetric):
    """Mock metric with different DIRECTIONS."""
    DIRECTIONS = {"recall": 1, "f1_score": 1}

    def _compute_score(self, y_pred, y_true):
        return {"recall": 0.8, "f1_score": 0.82}


class MockMetricOverlap(SingleTaskMetric):
    """Mock metric with overlapping DIRECTIONS."""
    DIRECTIONS = {"accuracy": 1, "auc": 1}  # "accuracy" overlaps with MockMetricA

    def _compute_score(self, y_pred, y_true):
        return {"accuracy": 0.88, "auc": 0.95}


def test_hybrid_metric_no_direction_overlap():
    """Test HybridMetric works when there are no DIRECTIONS key overlaps."""
    metrics_cfg = [
        {
            'class': MockMetricA,
            'args': {}
        },
        {
            'class': MockMetricB,
            'args': {}
        }
    ]

    # Should succeed - no overlapping keys
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Verify DIRECTIONS are correctly merged
    expected_directions = {
        "accuracy": 1, "precision": 1,  # From MockMetricA
        "recall": 1, "f1_score": 1      # From MockMetricB
    }
    assert hybrid_metric.DIRECTIONS == expected_directions


def test_hybrid_metric_direction_overlap_fails():
    """Test HybridMetric fails when there are DIRECTIONS key overlaps."""
    metrics_cfg = [
        {
            'class': MockMetricA,
            'args': {}
        },
        {
            'class': MockMetricOverlap,  # Has overlapping "accuracy" key
            'args': {}
        }
    ]

    # Should fail due to overlapping "accuracy" key in DIRECTIONS
    with pytest.raises(AssertionError, match="DIRECTIONS key overlap detected"):
        HybridMetric(metrics_cfg=metrics_cfg)


def test_hybrid_metric_single_component():
    """Test HybridMetric with single component (no overlap possible)."""
    metrics_cfg = [
        {
            'class': MockMetricA,
            'args': {}
        }
    ]

    # Should succeed - only one metric, no overlaps possible
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Should preserve the single metric's DIRECTIONS
    assert hybrid_metric.DIRECTIONS == MockMetricA.DIRECTIONS


def test_hybrid_metric_three_components_no_overlap():
    """Test HybridMetric with three components, all unique keys."""
    class MockMetricC(SingleTaskMetric):
        DIRECTIONS = {"mse": -1, "mae": -1}  # Different keys, different directions
        def _compute_score(self, y_pred, y_true):
            return {"mse": 0.1, "mae": 0.05}

    metrics_cfg = [
        {'class': MockMetricA, 'args': {}},
        {'class': MockMetricB, 'args': {}},
        {'class': MockMetricC, 'args': {}}
    ]

    # Should succeed - all keys are unique
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    expected_directions = {
        "accuracy": 1, "precision": 1,   # From MockMetricA
        "recall": 1, "f1_score": 1,      # From MockMetricB
        "mse": -1, "mae": -1             # From MockMetricC
    }
    assert hybrid_metric.DIRECTIONS == expected_directions


def test_hybrid_metric_three_components_with_overlap():
    """Test HybridMetric fails with three components where later ones overlap."""
    class MockMetricConflict(SingleTaskMetric):
        DIRECTIONS = {"precision": 1, "specificity": 1}  # "precision" conflicts with MockMetricA
        def _compute_score(self, y_pred, y_true):
            return {"precision": 0.9, "specificity": 0.88}

    metrics_cfg = [
        {'class': MockMetricA, 'args': {}},           # Has "precision"
        {'class': MockMetricB, 'args': {}},           # No conflict
        {'class': MockMetricConflict, 'args': {}}     # Also has "precision" -> conflict!
    ]

    # Should fail when processing MockMetricConflict due to "precision" overlap
    with pytest.raises(AssertionError, match="DIRECTIONS key overlap detected"):
        HybridMetric(metrics_cfg=metrics_cfg)