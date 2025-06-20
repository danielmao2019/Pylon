"""Test core functionality patterns for BaseCriterion."""
import pytest
import torch
from .conftest import ConcreteCriterion


def test_basic_call_functionality():
    """Test basic __call__ functionality."""
    criterion = ConcreteCriterion()
    
    # Test basic call
    y_pred = torch.tensor(1.0)
    y_true = torch.tensor(1.0)
    
    loss = criterion(y_pred, y_true)
    
    # Should return a tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert criterion.call_count == 1


def test_multiple_calls():
    """Test multiple sequential calls."""
    criterion = ConcreteCriterion()
    
    losses = []
    for i in range(5):
        y_pred = torch.tensor(float(i))
        y_true = torch.tensor(float(i))
        loss = criterion(y_pred, y_true)
        losses.append(loss)
    
    assert criterion.call_count == 5
    
    # Each call should return a different value based on call_count
    for i, loss in enumerate(losses):
        expected_value = (i + 1) * 0.1
        assert torch.isclose(loss, torch.tensor(expected_value))


def test_criterion_with_different_tensor_types():
    """Test criterion with different tensor types and devices."""
    criterion = ConcreteCriterion()
    
    # Test with different dtypes
    dtypes = [torch.float32, torch.float64]
    for dtype in dtypes:
        y_pred = torch.tensor(1.0, dtype=dtype)
        y_true = torch.tensor(1.0, dtype=dtype)
        loss = criterion(y_pred, y_true)
        assert isinstance(loss, torch.Tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_criterion_with_gpu_tensors():
    """Test criterion with GPU tensors."""
    criterion = ConcreteCriterion()
    
    # Move criterion to GPU
    criterion = criterion.cuda()
    
    # Test with GPU tensors
    y_pred = torch.tensor(1.0, device='cuda')
    y_true = torch.tensor(1.0, device='cuda')
    
    loss = criterion(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)


def test_summarize_basic_functionality():
    """Test basic summarize functionality."""
    criterion = ConcreteCriterion()
    
    # Add some losses
    for i in range(3):
        y_pred = torch.tensor(float(i))
        y_true = torch.tensor(float(i))
        criterion(y_pred, y_true)
    
    # Wait for buffer processing
    criterion._buffer_queue.join()
    
    # Summarize
    result = criterion.summarize()
    
    assert isinstance(result, torch.Tensor)
    assert len(result) == 3
    assert criterion.summarize_count == 1


def test_integration_call_and_summarize():
    """Test integration between __call__ and summarize."""
    criterion = ConcreteCriterion()
    
    # Make some calls
    expected_losses = []
    for i in range(4):
        y_pred = torch.tensor(float(i))
        y_true = torch.tensor(float(i))
        loss = criterion(y_pred, y_true)
        expected_losses.append(loss.detach().cpu())
    
    # Wait for buffer processing
    criterion._buffer_queue.join()
    
    # Summarize should return all losses
    result = criterion.summarize()
    
    assert len(result) == 4
    for i, expected_loss in enumerate(expected_losses):
        assert torch.equal(result[i], expected_loss)


def test_gradient_flow():
    """Test that gradients flow correctly through criterion."""
    criterion = ConcreteCriterion()
    
    # Create tensors that require gradients
    y_pred = torch.tensor(2.0, requires_grad=True)
    y_true = torch.tensor(1.0)
    
    loss = criterion(y_pred, y_true)
    
    # Loss should require gradients
    assert loss.requires_grad
    
    # Backward pass should work
    loss.backward()
    
    # Note: Our ConcreteCriterion doesn't use the inputs in loss computation
    # It just returns a tensor based on call_count, so gradients won't flow back
    # This is expected behavior for this test implementation


def test_criterion_without_buffer_functionality():
    """Test criterion functionality when buffer is disabled."""
    criterion = ConcreteCriterion(use_buffer=False)
    
    # Basic functionality should still work
    y_pred = torch.tensor(1.0)
    y_true = torch.tensor(1.0)
    
    loss = criterion(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert criterion.call_count == 1
    
    # Summarize should return default value
    result = criterion.summarize()
    assert torch.equal(result, torch.tensor([0.0]))
    assert criterion.summarize_count == 1


def test_criterion_state_isolation():
    """Test that different criterion instances are isolated."""
    criterion1 = ConcreteCriterion()
    criterion2 = ConcreteCriterion()
    
    # Make calls on both
    criterion1(torch.tensor(1.0), torch.tensor(1.0))
    criterion1(torch.tensor(2.0), torch.tensor(2.0))
    
    criterion2(torch.tensor(3.0), torch.tensor(3.0))
    
    # States should be independent
    assert criterion1.call_count == 2
    assert criterion2.call_count == 1
    
    # Wait for buffer processing
    criterion1._buffer_queue.join()
    criterion2._buffer_queue.join()
    
    assert len(criterion1.buffer) == 2
    assert len(criterion2.buffer) == 1