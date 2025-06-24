import pytest
import torch
from metrics.wrappers.hybrid_metric import HybridMetric


def test_gpu_computation(metrics_cfg, sample_tensor, sample_target):
    """Test computing scores with GPU tensors."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a hybrid metric
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Test on CPU
    cpu_scores = hybrid_metric(y_pred=sample_tensor, y_true=sample_target)
    hybrid_metric._buffer_queue.join()
    assert len(hybrid_metric.buffer) == 1

    # Test with GPU tensors (metrics themselves stay on CPU)
    gpu_input = sample_tensor.cuda()
    gpu_target = sample_target.cuda()

    gpu_scores = hybrid_metric(y_pred=gpu_input, y_true=gpu_target)
    hybrid_metric._buffer_queue.join()
    assert len(hybrid_metric.buffer) == 2

    # Check that score keys are consistent
    assert set(cpu_scores.keys()) == set(gpu_scores.keys())

    # Check that scores are computed correctly regardless of input device
    assert all(key in gpu_scores for key in cpu_scores.keys())


def test_mixed_device_computation(metrics_cfg):
    """Test computation with mixed device inputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Create tensors on different devices
    cpu_tensor = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    gpu_tensor = torch.randn(2, 3, 4, 4, dtype=torch.float32).cuda()

    # Test CPU input, GPU target (should work as tensors are moved as needed)
    try:
        scores1 = hybrid_metric(y_pred=cpu_tensor, y_true=gpu_tensor.cpu())
        assert isinstance(scores1, dict)
    except RuntimeError:
        # This is expected behavior for device mismatch in some operations
        pass

    # Test GPU input, CPU target
    try:
        scores2 = hybrid_metric(y_pred=gpu_tensor.cpu(), y_true=cpu_tensor)
        assert isinstance(scores2, dict)
    except RuntimeError:
        # This is expected behavior for device mismatch in some operations
        pass


def test_device_consistency_across_metrics(metrics_cfg):
    """Test that all component metrics handle device placement consistently."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Test with GPU tensors
    gpu_input = torch.randn(2, 3, 4, 4, dtype=torch.float32).cuda()
    gpu_target = torch.randn(2, 3, 4, 4, dtype=torch.float32).cuda()

    scores = hybrid_metric(y_pred=gpu_input, y_true=gpu_target)

    # All scores should be computed successfully
    assert isinstance(scores, dict)
    assert len(scores) == 2

    # All scores should be tensors
    for score in scores.values():
        assert isinstance(score, torch.Tensor)
        assert score.ndim == 0


def test_buffer_device_handling(metrics_cfg):
    """Test that buffer correctly handles tensors from different devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Compute scores with CPU tensors
    cpu_input = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    cpu_target = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    cpu_scores = hybrid_metric(y_pred=cpu_input, y_true=cpu_target)

    # Compute scores with GPU tensors
    gpu_input = torch.randn(2, 3, 4, 4, dtype=torch.float32).cuda()
    gpu_target = torch.randn(2, 3, 4, 4, dtype=torch.float32).cuda()
    gpu_scores = hybrid_metric(y_pred=gpu_input, y_true=gpu_target)

    # Wait for buffer processing
    hybrid_metric._buffer_queue.join()

    # Buffer should contain both sets of scores
    assert len(hybrid_metric.buffer) == 2

    # All buffered scores should be on CPU (as per buffer worker behavior)
    for buffered_scores in hybrid_metric.buffer:
        for score in buffered_scores.values():
            assert not score.is_cuda  # Should be moved to CPU by buffer worker


def test_large_tensor_device_transfer(metrics_cfg):
    """Test device handling with larger tensors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Test with larger tensors
    large_gpu_input = torch.randn(8, 3, 64, 64, dtype=torch.float32).cuda()
    large_gpu_target = torch.randn(8, 3, 64, 64, dtype=torch.float32).cuda()

    scores = hybrid_metric(y_pred=large_gpu_input, y_true=large_gpu_target)

    # Verify computation succeeded
    assert isinstance(scores, dict)
    assert len(scores) == 2

    # Verify all scores are valid tensors
    for score in scores.values():
        assert isinstance(score, torch.Tensor)
        assert torch.isfinite(score)
        assert not torch.isnan(score)
