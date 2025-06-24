import pytest
import torch
from criteria.wrappers.hybrid_criterion import HybridCriterion


def test_device_transfer(criteria_cfg, sample_tensor):
    """Test moving the criterion between CPU and GPU."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a criterion
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)

    # Step 1: Test on CPU
    # Check initial state
    for component_criterion in criterion.criteria:
        assert not component_criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 0

    # Compute loss on CPU
    y_true = torch.randn_like(sample_tensor)
    cpu_loss = criterion(y_pred=sample_tensor, y_true=y_true)
    criterion._buffer_queue.join()
    assert len(criterion.buffer) == 1

    # Step 2: Move to GPU
    criterion = criterion.cuda()
    gpu_input = sample_tensor.cuda()
    gpu_target = y_true.cuda()

    # Check GPU state
    for component_criterion in criterion.criteria:
        assert component_criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 1

    # Compute loss on GPU
    gpu_loss = criterion(y_pred=gpu_input, y_true=gpu_target)
    criterion._buffer_queue.join()
    assert len(criterion.buffer) == 2

    # Step 3: Move back to CPU
    criterion = criterion.cpu()

    # Check CPU state
    for component_criterion in criterion.criteria:
        assert not component_criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 2

    # Compute loss on CPU again
    cpu_loss2 = criterion(y_pred=sample_tensor, y_true=y_true)
    criterion._buffer_queue.join()
    assert len(criterion.buffer) == 3

    # Check that all losses are equivalent
    assert abs(cpu_loss.item() - gpu_loss.item()) < 1e-5
    assert abs(cpu_loss.item() - cpu_loss2.item()) < 1e-5


def test_gpu_computation(criteria_cfg, sample_tensor):
    """Test computing losses with GPU tensors."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create a criterion (stays on CPU)
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    
    # Test on CPU
    cpu_target = torch.randn_like(sample_tensor)
    cpu_loss = criterion(y_pred=sample_tensor, y_true=cpu_target)
    criterion._buffer_queue.join()
    assert len(criterion.buffer) == 1
    
    # Move criterion to GPU and test with GPU tensors
    criterion = criterion.cuda()
    gpu_input = sample_tensor.cuda()
    gpu_target = cpu_target.cuda()
    
    gpu_loss = criterion(y_pred=gpu_input, y_true=gpu_target)
    criterion._buffer_queue.join()
    assert len(criterion.buffer) == 2
    
    # Check that both losses are valid
    assert torch.isfinite(cpu_loss)
    assert torch.isfinite(gpu_loss)
    
    # Check that results are similar (should be identical for same inputs)
    assert abs(cpu_loss.item() - gpu_loss.item()) < 1e-5


def test_mixed_device_computation(criteria_cfg):
    """Test computation with mixed device inputs (should fail gracefully or work)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    
    # Create tensors on different devices
    cpu_tensor = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    gpu_tensor = torch.randn(2, 3, 4, 4, dtype=torch.float32).cuda()
    
    # Test CPU criterion with mixed inputs (should raise RuntimeError)
    with pytest.raises(RuntimeError):
        criterion(y_pred=cpu_tensor, y_true=gpu_tensor)


def test_buffer_device_handling(criteria_cfg, sample_tensor):
    """Test that buffer correctly handles tensors from different devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    
    # Compute losses with CPU tensors
    cpu_target = torch.randn_like(sample_tensor)
    cpu_loss = criterion(y_pred=sample_tensor, y_true=cpu_target)
    
    # Move to GPU and compute losses with GPU tensors
    criterion = criterion.cuda()
    gpu_input = sample_tensor.cuda()
    gpu_target = cpu_target.cuda()
    gpu_loss = criterion(y_pred=gpu_input, y_true=gpu_target)
    
    # Wait for buffer processing
    criterion._buffer_queue.join()
    
    # Buffer should contain both losses
    assert len(criterion.buffer) == 2
    
    # All buffered losses should be on CPU (as per buffer worker behavior)
    for buffered_loss in criterion.buffer:
        assert not buffered_loss.is_cuda  # Should be moved to CPU by buffer worker


def test_large_tensor_device_transfer(criteria_cfg):
    """Test device handling with larger tensors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    
    # Test with larger tensors
    large_gpu_input = torch.randn(8, 3, 64, 64, dtype=torch.float32).cuda()
    large_gpu_target = torch.randn(8, 3, 64, 64, dtype=torch.float32).cuda()
    
    # Move criterion to GPU
    criterion = criterion.cuda()
    
    loss = criterion(y_pred=large_gpu_input, y_true=large_gpu_target)
    
    # Verify computation succeeded
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    
    # Verify loss is valid
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
