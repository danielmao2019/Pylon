import pytest
import torch
from metrics.wrappers.hybrid_metric import HybridMetric


def test_empty_metrics_config():
    """Test that empty metrics config raises assertion error."""
    with pytest.raises(AssertionError):
        HybridMetric(metrics_cfg=[])

    with pytest.raises(AssertionError):
        HybridMetric(metrics_cfg=None)


def test_no_key_overlap_assertion(sample_tensor, sample_target, dummy_metric, another_dummy_metric):
    """Test that the metric properly detects key overlaps."""
    # Create metrics with overlapping keys
    overlapping_metrics_cfg = [
        {
            'class': dummy_metric.__class__,
            'args': {
                'metric_name': 'same_name',
            }
        },
        {
            'class': another_dummy_metric.__class__,
            'args': {
                'metric_name': 'same_name',  # Same name - should cause overlap
            }
        }
    ]

    hybrid_metric = HybridMetric(metrics_cfg=overlapping_metrics_cfg)

    # This should raise an assertion error due to key overlap
    with pytest.raises(AssertionError, match="Key overlap detected"):
        hybrid_metric(y_pred=sample_tensor, y_true=sample_target)


def test_invalid_metric_config():
    """Test that invalid metric configurations raise appropriate errors."""
    # Test with invalid class
    with pytest.raises(Exception):  # Could be various exception types depending on build_from_config
        invalid_cfg = [
            {
                'class': 'NotAClass',
                'args': {}
            }
        ]
        HybridMetric(metrics_cfg=invalid_cfg)


def test_metric_build_failure(dummy_metric):
    """Test handling of metric build failures."""
    # Test with missing required arguments
    with pytest.raises(Exception):
        invalid_cfg = [
            {
                'class': dummy_metric.__class__,
                'args': {
                    'invalid_arg': 'value'  # Missing required metric_name
                }
            }
        ]
        # This should fail during metric creation
        hybrid_metric = HybridMetric(metrics_cfg=invalid_cfg)


def test_tensor_input_validation(metrics_cfg):
    """Test that invalid tensor inputs are handled appropriately."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Test with mismatched tensor shapes
    sample_input = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    mismatched_target = torch.randn(2, 3, 8, 8, dtype=torch.float32)  # Different shape

    # This should raise an error from the underlying metric computation
    with pytest.raises(RuntimeError):
        hybrid_metric(y_pred=sample_input, y_true=mismatched_target)


def test_non_tensor_inputs(metrics_cfg):
    """Test that non-tensor inputs are handled appropriately."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Test with non-tensor inputs
    with pytest.raises(Exception):
        hybrid_metric(y_pred="not_a_tensor", y_true="also_not_a_tensor")


def test_complex_key_overlap_scenarios(dummy_metric, another_dummy_metric):
    """Test complex scenarios for key overlap detection."""
    # Test partial overlap with multiple metrics
    complex_overlapping_cfg = [
        {
            'class': dummy_metric.__class__,
            'args': {'metric_name': 'unique1'}
        },
        {
            'class': another_dummy_metric.__class__,
            'args': {'metric_name': 'shared'}
        },
        {
            'class': dummy_metric.__class__,
            'args': {'metric_name': 'shared'}  # This creates overlap
        }
    ]

    hybrid_metric = HybridMetric(metrics_cfg=complex_overlapping_cfg)

    sample_input = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    sample_target = torch.randn(2, 3, 4, 4, dtype=torch.float32)

    with pytest.raises(AssertionError, match="Key overlap detected"):
        hybrid_metric(y_pred=sample_input, y_true=sample_target)
