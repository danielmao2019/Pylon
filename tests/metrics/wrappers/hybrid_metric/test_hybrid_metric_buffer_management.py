import pytest
import torch
from metrics.wrappers.hybrid_metric import HybridMetric


def create_datapoint(y_pred, y_true, idx=0):
    """Helper function to create datapoint from y_pred and y_true."""
    return {
        'inputs': {},  # Empty for these tests
        'outputs': y_pred,
        'labels': y_true,
        'meta_info': {'idx': idx}
    }


def test_buffer_behavior(metrics_cfg, sample_tensor, sample_target):
    """Test the buffer behavior of HybridMetric."""
    # Create a hybrid metric
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Test initialize
    assert hybrid_metric.use_buffer is True
    assert hasattr(hybrid_metric, 'buffer') and hybrid_metric.buffer == {}
    for component_metric in hybrid_metric.metrics:
        assert component_metric.use_buffer is False
        assert not hasattr(component_metric, 'buffer')

    # Test update
    datapoint = create_datapoint(sample_tensor, sample_target)
    scores = hybrid_metric(datapoint)
    hybrid_metric._buffer_queue.join()
    assert hybrid_metric.use_buffer is True
    assert hasattr(hybrid_metric, 'buffer') and len(hybrid_metric.buffer) == 1
    assert 0 in hybrid_metric.buffer  # Index 0 should be in buffer
    # Compare scores stored in buffer (note: buffer contains CPU versions)
    buffer_scores = hybrid_metric.buffer[0]
    for key in scores.keys():
        assert torch.allclose(scores[key].cpu(), buffer_scores[key])
    for component_metric in hybrid_metric.metrics:
        assert component_metric.use_buffer is False
        assert not hasattr(component_metric, 'buffer')

    # Test reset
    hybrid_metric.reset_buffer()
    assert hybrid_metric.use_buffer is True
    assert hasattr(hybrid_metric, 'buffer') and hybrid_metric.buffer == {}
    for component_metric in hybrid_metric.metrics:
        assert component_metric.use_buffer is False
        assert not hasattr(component_metric, 'buffer')


def test_disabled_buffer_initialization(metrics_cfg, sample_tensor, sample_target):
    """Test HybridMetric with disabled buffer."""
    # Create a hybrid metric with disabled buffer
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg, use_buffer=False)

    # Test initialization
    assert hybrid_metric.use_buffer is False
    assert not hasattr(hybrid_metric, 'buffer')
    for component_metric in hybrid_metric.metrics:
        assert component_metric.use_buffer is False
        assert not hasattr(component_metric, 'buffer')

    # Test that scores are still computed correctly
    datapoint = create_datapoint(sample_tensor, sample_target)
    scores = hybrid_metric(datapoint)
    assert isinstance(scores, dict)
    assert 'metric1' in scores
    assert 'metric2' in scores

    # Test that summarize raises error when buffer is disabled
    with pytest.raises(AssertionError):
        hybrid_metric.summarize()


def test_component_metrics_force_disabled_buffer(sample_tensor, sample_target, dummy_metric, another_dummy_metric):
    """Test that component metrics have their buffers forcibly disabled."""
    # Create configs where metrics initially have use_buffer=True
    metrics_cfg_with_buffer = [
        {
            'class': dummy_metric.__class__,
            'args': {
                'metric_name': 'metric1',
                'use_buffer': True,  # This should be overridden
            }
        },
        {
            'class': another_dummy_metric.__class__,
            'args': {
                'metric_name': 'metric2',
                'use_buffer': True,  # This should be overridden
            }
        }
    ]

    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg_with_buffer)

    # Verify that component metrics have use_buffer=False despite initial config
    for component_metric in hybrid_metric.metrics:
        assert component_metric.use_buffer is False
        assert not hasattr(component_metric, 'buffer')


def test_buffer_operations_thread_safety(metrics_cfg, sample_tensor, sample_target):
    """Test that buffer operations are thread-safe."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Perform multiple operations to test thread safety
    for i in range(5):
        datapoint = create_datapoint(sample_tensor, sample_target, idx=i)
        scores = hybrid_metric(datapoint)
        assert isinstance(scores, dict)

    # Wait for all operations to complete
    hybrid_metric._buffer_queue.join()

    # Verify buffer contains all scores
    assert len(hybrid_metric.buffer) == 5

    # Test that buffer can be safely accessed
    buffer_copy = hybrid_metric.get_buffer()
    assert len(buffer_copy) == 5
    assert all(isinstance(item, dict) for item in buffer_copy)
