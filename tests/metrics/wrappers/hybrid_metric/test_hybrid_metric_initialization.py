import pytest
import torch
from metrics.wrappers.hybrid_metric import HybridMetric


def test_basic_initialization(metrics_cfg):
    """Test that the metrics are properly initialized."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Test that the metrics are properly stored
    assert hasattr(hybrid_metric, 'metrics')
    assert isinstance(hybrid_metric.metrics, list)
    assert len(hybrid_metric.metrics) == 2

    # Test that each metric is properly created
    assert hybrid_metric.metrics[0].__class__.__name__ == 'DummyMetric'
    assert hybrid_metric.metrics[1].__class__.__name__ == 'AnotherDummyMetric'


def test_inheritance(dummy_metric):
    """Test that HybridMetric properly inherits from SingleTaskMetric."""
    from metrics.wrappers.single_task_metric import SingleTaskMetric

    hybrid_metric = HybridMetric(metrics_cfg=[
        {'class': dummy_metric.__class__, 'args': {'metric_name': 'test'}}
    ])

    assert isinstance(hybrid_metric, SingleTaskMetric)
    # Test that it has the inherited methods from SingleTaskMetric
    assert hasattr(hybrid_metric, 'summarize')
    assert callable(hybrid_metric.summarize)


def test_base_metric_inheritance(dummy_metric):
    """Test that HybridMetric inherits from BaseMetric."""
    from metrics.base_metric import BaseMetric

    hybrid_metric = HybridMetric(metrics_cfg=[
        {'class': dummy_metric.__class__, 'args': {'metric_name': 'test'}}
    ])

    assert isinstance(hybrid_metric, BaseMetric)
    # Test that it has the base metric attributes and methods
    assert hasattr(hybrid_metric, 'use_buffer')
    assert hasattr(hybrid_metric, 'add_to_buffer')
    assert hasattr(hybrid_metric, 'reset_buffer')


def test_default_buffer_enabled(metrics_cfg):
    """Test that buffer is enabled by default."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    assert hybrid_metric.use_buffer is True
    assert hasattr(hybrid_metric, 'buffer')
    assert isinstance(hybrid_metric.buffer, list)
    assert len(hybrid_metric.buffer) == 0


def test_explicit_buffer_configuration(metrics_cfg):
    """Test explicit buffer configuration."""
    # Test with buffer explicitly enabled
    hybrid_metric_enabled = HybridMetric(metrics_cfg=metrics_cfg, use_buffer=True)
    assert hybrid_metric_enabled.use_buffer is True

    # Test with buffer explicitly disabled
    hybrid_metric_disabled = HybridMetric(metrics_cfg=metrics_cfg, use_buffer=False)
    assert hybrid_metric_disabled.use_buffer is False


def test_component_metric_types(metrics_cfg):
    """Test that all component metrics are BaseMetric instances."""
    from metrics.base_metric import BaseMetric

    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    for metric in hybrid_metric.metrics:
        assert isinstance(metric, BaseMetric)
