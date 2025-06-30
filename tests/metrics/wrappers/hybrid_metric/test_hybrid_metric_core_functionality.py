import pytest
import torch
from metrics.wrappers.hybrid_metric import HybridMetric


def test_compute_score_merge(metrics_cfg, sample_tensor, sample_target):
    """Test computing merged scores from multiple metrics."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Compute scores using the hybrid metric
    scores = hybrid_metric(y_pred=sample_tensor, y_true=sample_target)

    # Check that we get scores from both metrics
    assert isinstance(scores, dict)
    assert 'metric1' in scores
    assert 'metric2' in scores
    assert len(scores) == 2

    # Check that each score is a tensor
    assert isinstance(scores['metric1'], torch.Tensor)
    assert isinstance(scores['metric2'], torch.Tensor)
    assert scores['metric1'].ndim == 0  # scalar
    assert scores['metric2'].ndim == 0  # scalar


def test_score_computation_consistency(metrics_cfg, sample_tensor, sample_target):
    """Test that scores are computed consistently across multiple calls."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Compute scores multiple times with same inputs
    scores1 = hybrid_metric(y_pred=sample_tensor, y_true=sample_target)
    scores2 = hybrid_metric(y_pred=sample_tensor, y_true=sample_target)

    # Scores should have same keys
    assert set(scores1.keys()) == set(scores2.keys())

    # Individual scores should be consistent (since inputs are deterministic)
    for key in scores1.keys():
        assert torch.allclose(scores1[key], scores2[key])


def test_different_input_shapes(metrics_cfg):
    """Test that hybrid metric works with different input shapes."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Test with different tensor shapes
    shapes = [(1, 3, 4, 4), (2, 3, 8, 8), (4, 3, 16, 16)]

    for shape in shapes:
        sample_input = torch.randn(shape, dtype=torch.float32)
        sample_target = torch.randn(shape, dtype=torch.float32)

        scores = hybrid_metric(y_pred=sample_input, y_true=sample_target)

        # Verify scores are computed correctly
        assert isinstance(scores, dict)
        assert len(scores) == 2
        assert all(isinstance(score, torch.Tensor) for score in scores.values())
        assert all(score.ndim == 0 for score in scores.values())


def test_single_metric_configuration(dummy_metric):
    """Test hybrid metric with only one component metric."""
    single_metric_cfg = [
        {
            'class': dummy_metric.__class__,
            'args': {
                'metric_name': 'single_metric',
            }
        }
    ]

    hybrid_metric = HybridMetric(metrics_cfg=single_metric_cfg)

    sample_input = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    sample_target = torch.randn(2, 3, 4, 4, dtype=torch.float32)

    scores = hybrid_metric(y_pred=sample_input, y_true=sample_target)

    assert isinstance(scores, dict)
    assert len(scores) == 1
    assert 'single_metric' in scores
    assert isinstance(scores['single_metric'], torch.Tensor)


def test_multiple_metrics_configuration(dummy_metric, another_dummy_metric):
    """Test hybrid metric with multiple component metrics."""
    multi_metric_cfg = [
        {
            'class': dummy_metric.__class__,
            'args': {'metric_name': 'metric_a'}
        },
        {
            'class': another_dummy_metric.__class__,
            'args': {'metric_name': 'metric_b'}
        },
        {
            'class': dummy_metric.__class__,
            'args': {'metric_name': 'metric_c'}
        }
    ]

    hybrid_metric = HybridMetric(metrics_cfg=multi_metric_cfg)

    sample_input = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    sample_target = torch.randn(2, 3, 4, 4, dtype=torch.float32)

    scores = hybrid_metric(y_pred=sample_input, y_true=sample_target)

    assert isinstance(scores, dict)
    assert len(scores) == 3
    assert 'metric_a' in scores
    assert 'metric_b' in scores
    assert 'metric_c' in scores
    assert all(isinstance(score, torch.Tensor) for score in scores.values())


def test_compute_score_method_directly(metrics_cfg, sample_tensor, sample_target):
    """Test the _compute_score method directly."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)
    
    # Call _compute_score directly
    scores = hybrid_metric._compute_score(y_pred=sample_tensor, y_true=sample_target)
    
    # Verify it returns the expected format
    assert isinstance(scores, dict)
    assert 'metric1' in scores
    assert 'metric2' in scores
    assert len(scores) == 2
    
    # Verify tensor properties
    for score in scores.values():
        assert isinstance(score, torch.Tensor)
        assert score.ndim == 0  # scalar
        assert torch.isfinite(score)
        assert not torch.isnan(score)


def test_score_tensor_properties(metrics_cfg, sample_tensor, sample_target):
    """Test that scores have correct tensor properties."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)
    
    scores = hybrid_metric(y_pred=sample_tensor, y_true=sample_target)
    
    for key, score in scores.items():
        # Basic tensor properties
        assert isinstance(score, torch.Tensor)
        assert score.dtype == torch.float32
        assert score.ndim == 0  # scalar
        assert score.requires_grad is False  # Should be detached from computation graph
        
        # Numerical properties
        assert torch.isfinite(score), f"Score {key} is not finite: {score}"
        assert not torch.isnan(score), f"Score {key} is NaN: {score}"
        assert not torch.isinf(score), f"Score {key} is infinite: {score}"


def test_component_metric_isolation(dummy_metric, another_dummy_metric):
    """Test that component metrics are properly isolated."""
    metrics_cfg = [
        {
            'class': dummy_metric.__class__,
            'args': {'metric_name': 'isolated1'}
        },
        {
            'class': another_dummy_metric.__class__,
            'args': {'metric_name': 'isolated2'}
        }
    ]
    
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)
    
    # Verify each component metric is independent
    assert len(hybrid_metric.metrics) == 2
    assert hybrid_metric.metrics[0] is not hybrid_metric.metrics[1]
    
    # Verify they don't share state
    sample_input = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    sample_target = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    
    scores = hybrid_metric(y_pred=sample_input, y_true=sample_target)
    
    # Each should produce different scores (since they use different computations)
    assert scores['isolated1'] != scores['isolated2']


def test_deterministic_computation(metrics_cfg):
    """Test that computation is deterministic with same inputs."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)
    
    # Use fixed seed for deterministic inputs
    torch.manual_seed(42)
    sample_input = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    sample_target = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    
    # Compute scores multiple times
    scores1 = hybrid_metric(y_pred=sample_input, y_true=sample_target)
    scores2 = hybrid_metric(y_pred=sample_input, y_true=sample_target)
    
    # Should be identical
    for key in scores1.keys():
        assert torch.equal(scores1[key], scores2[key]), f"Non-deterministic computation for {key}"
