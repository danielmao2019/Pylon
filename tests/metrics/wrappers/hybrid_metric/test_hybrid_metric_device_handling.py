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


def test_gpu_computation(metrics_cfg, sample_tensor, sample_target):
    """Test computing scores with GPU tensors."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a hybrid metric
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Test on CPU
    datapoint = create_datapoint(sample_tensor, sample_target, idx=0)
    cpu_scores = hybrid_metric(datapoint)
    hybrid_metric._buffer_queue.join()
    assert len(hybrid_metric.buffer) == 1

    # Test with GPU tensors (metrics themselves stay on CPU)
    gpu_input = sample_tensor.cuda()
    gpu_target = sample_target.cuda()

    datapoint = create_datapoint(gpu_input, gpu_target, idx=1)
    gpu_scores = hybrid_metric(datapoint)
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

    # Test CPU input, GPU target - expect RuntimeError for device mismatch
    with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
        datapoint = create_datapoint(cpu_tensor, gpu_tensor)
        hybrid_metric(datapoint)

    # Test GPU input, CPU target - expect RuntimeError for device mismatch  
    with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
        datapoint = create_datapoint(gpu_tensor, cpu_tensor)
        hybrid_metric(datapoint)


def test_device_consistency_across_metrics(metrics_cfg):
    """Test that all component metrics handle device placement consistently."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Test with GPU tensors
    gpu_input = torch.randn(2, 3, 4, 4, dtype=torch.float32).cuda()
    gpu_target = torch.randn(2, 3, 4, 4, dtype=torch.float32).cuda()

    datapoint = create_datapoint(gpu_input, gpu_target)
    scores = hybrid_metric(datapoint)

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
    datapoint = create_datapoint(cpu_input, cpu_target, idx=0)
    _ = hybrid_metric(datapoint)

    # Compute scores with GPU tensors
    gpu_input = torch.randn(2, 3, 4, 4, dtype=torch.float32).cuda()
    gpu_target = torch.randn(2, 3, 4, 4, dtype=torch.float32).cuda()
    datapoint = create_datapoint(gpu_input, gpu_target, idx=1)
    _ = hybrid_metric(datapoint)

    # Wait for buffer processing
    hybrid_metric._buffer_queue.join()

    # Buffer should contain both sets of scores
    assert len(hybrid_metric.buffer) == 2

    # All buffered scores should be on CPU (as per buffer worker behavior)
    for idx in hybrid_metric.buffer:
        buffered_scores = hybrid_metric.buffer[idx]
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

    datapoint = create_datapoint(large_gpu_input, large_gpu_target)
    scores = hybrid_metric(datapoint)

    # Verify computation succeeded
    assert isinstance(scores, dict)
    assert len(scores) == 2

    # Verify all scores are valid tensors
    for score in scores.values():
        assert isinstance(score, torch.Tensor)
        assert torch.isfinite(score)
        assert not torch.isnan(score)
