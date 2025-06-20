"""Test device transfer patterns for BaseCriterion."""
import pytest
import torch
from .conftest import ConcreteCriterion


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_transfer_with_buffer():
    """Test moving criterion between CPU and GPU with buffer."""
    criterion = ConcreteCriterion()
    
    # Add some data on CPU
    criterion.add_to_buffer(torch.tensor(1.0))
    criterion._buffer_queue.join()
    
    # Move to GPU
    criterion = criterion.cuda()
    
    # Buffer should remain on CPU (buffers are always CPU)
    assert not criterion.buffer[0].is_cuda
    
    # Add GPU data
    gpu_tensor = torch.tensor(2.0, device='cuda')
    criterion.add_to_buffer(gpu_tensor)
    criterion._buffer_queue.join()
    
    # Buffer items should be moved to CPU
    assert not criterion.buffer[1].is_cuda
    assert torch.equal(criterion.buffer[1], gpu_tensor.detach().cpu())
    
    # Move back to CPU
    criterion = criterion.cpu()
    
    # Buffer should still work
    criterion.add_to_buffer(torch.tensor(3.0))
    criterion._buffer_queue.join()
    
    assert len(criterion.buffer) == 3
    assert all(not item.is_cuda for item in criterion.buffer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_transfer_threading_preservation():
    """Test that threading infrastructure survives device transfer."""
    criterion = ConcreteCriterion()
    
    # Verify threading infrastructure exists
    original_thread = criterion._buffer_thread
    assert original_thread.is_alive()
    
    # Move to GPU
    criterion = criterion.cuda()
    
    # Threading infrastructure should be preserved
    assert hasattr(criterion, '_buffer_thread')
    assert hasattr(criterion, '_buffer_queue')
    assert hasattr(criterion, '_buffer_lock')
    assert criterion._buffer_thread.is_alive()
    
    # Should be able to add data
    criterion.add_to_buffer(torch.tensor(1.0, device='cuda'))
    criterion._buffer_queue.join()
    
    assert len(criterion.buffer) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_device_tensors():
    """Test handling tensors from different devices."""
    criterion = ConcreteCriterion()
    
    # Add tensors from different devices
    cpu_tensor = torch.tensor(1.0)
    gpu_tensor = torch.tensor(2.0, device='cuda')
    
    criterion.add_to_buffer(cpu_tensor)
    criterion.add_to_buffer(gpu_tensor)
    criterion._buffer_queue.join()
    
    # All buffer items should be on CPU
    assert len(criterion.buffer) == 2
    assert not criterion.buffer[0].is_cuda
    assert not criterion.buffer[1].is_cuda
    assert torch.equal(criterion.buffer[0], cpu_tensor)
    assert torch.equal(criterion.buffer[1], gpu_tensor.cpu())


def test_device_handling_without_buffer():
    """Test device handling when buffer is disabled."""
    criterion = ConcreteCriterion(use_buffer=False)
    
    # Should work fine with CPU
    dummy_data = torch.tensor(1.0)
    loss = criterion(dummy_data, dummy_data)
    assert loss.device == torch.device('cpu')
    
    # Move to GPU if available
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        gpu_data = torch.tensor(1.0, device='cuda')
        loss = criterion(gpu_data, gpu_data)
        # Loss computation happens in subclass, device depends on implementation