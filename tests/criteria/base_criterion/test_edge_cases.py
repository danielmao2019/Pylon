"""Test edge cases and error handling for BaseCriterion."""
import pytest
import torch
import tempfile
from unittest.mock import patch
from criteria.base_criterion import BaseCriterion
from .conftest import ConcreteCriterion


def test_abstract_method_enforcement():
    """Test that abstract methods must be implemented."""
    # Cannot instantiate BaseCriterion directly
    with pytest.raises(TypeError):
        BaseCriterion()
    
    # Incomplete implementation should fail
    class IncompleteCriterion(BaseCriterion):
        def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            pass
        # Missing summarize method
    
    with pytest.raises(TypeError):
        IncompleteCriterion()
    
    class AnotherIncompleteCriterion(BaseCriterion):
        def summarize(self, output_path=None):
            pass
        # Missing __call__ method
    
    with pytest.raises(TypeError):
        AnotherIncompleteCriterion()


def test_buffer_operations_with_buffer_disabled():
    """Test buffer operations when buffer is disabled."""
    criterion = ConcreteCriterion(use_buffer=False)
    
    # Should not have buffer-related attributes
    assert not hasattr(criterion, 'buffer')
    assert not hasattr(criterion, '_buffer_lock')
    assert not hasattr(criterion, '_buffer_queue')
    assert not hasattr(criterion, '_buffer_thread')
    
    # Adding to buffer should not fail - it just checks that buffer doesn't exist
    # The assertion is that buffer attribute doesn't exist, not that it should raise
    criterion.add_to_buffer(torch.tensor(1.0))  # Should pass the assertion check
    
    # Getting buffer should fail
    with pytest.raises(RuntimeError, match="Buffer is not enabled"):
        criterion.get_buffer()
    
    # Reset should work without issues
    criterion.reset_buffer()


def test_summarize_with_output_path():
    """Test summarize method with file output."""
    criterion = ConcreteCriterion()
    
    # Add some data
    for i in range(3):
        criterion.add_to_buffer(torch.tensor(float(i)))
    
    criterion._buffer_queue.join()
    
    # Test saving to file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        result = criterion.summarize(output_path=tmp_file.name)
        
        # Verify result
        assert len(result) == 3
        
        # Verify file was saved
        loaded_result = torch.load(tmp_file.name)
        assert torch.equal(result, loaded_result)


def test_summarize_empty_buffer():
    """Test summarize with empty buffer."""
    criterion = ConcreteCriterion()
    
    # No data added
    result = criterion.summarize()
    assert torch.equal(result, torch.tensor([0.0]))


def test_summarize_without_buffer():
    """Test summarize when buffer is disabled."""
    criterion = ConcreteCriterion(use_buffer=False)
    
    result = criterion.summarize()
    assert torch.equal(result, torch.tensor([0.0]))


def test_state_consistency_after_operations():
    """Test state consistency after various operations."""
    criterion = ConcreteCriterion()
    
    # Initial state
    assert len(criterion.buffer) == 0
    assert criterion.call_count == 0
    assert criterion.summarize_count == 0
    
    # After calling
    dummy_data = torch.tensor(1.0)
    loss = criterion(dummy_data, dummy_data)
    criterion._buffer_queue.join()
    
    assert criterion.call_count == 1
    assert len(criterion.buffer) == 1
    assert torch.equal(criterion.buffer[0], loss.detach().cpu())
    
    # After summarizing
    result = criterion.summarize()
    assert criterion.summarize_count == 1
    assert torch.equal(result[0], loss.detach().cpu())
    
    # After resetting
    criterion.reset_buffer()
    assert len(criterion.buffer) == 0
    assert criterion.call_count == 1  # Should preserve call count
    assert criterion.summarize_count == 1  # Should preserve summarize count


def test_state_preservation_across_multiple_calls():
    """Test state preservation across multiple calls."""
    criterion = ConcreteCriterion()
    
    losses = []
    for i in range(5):
        dummy_data = torch.tensor(float(i))
        loss = criterion(dummy_data, dummy_data)
        losses.append(loss)
    
    criterion._buffer_queue.join()
    
    # Verify all losses are preserved
    assert len(criterion.buffer) == 5
    assert criterion.call_count == 5
    
    for i, expected_loss in enumerate(losses):
        assert torch.equal(criterion.buffer[i], expected_loss.detach().cpu())


def test_reset_buffer_with_non_empty_queue():
    """Test reset buffer with items still in queue."""
    criterion = ConcreteCriterion()
    
    # Add data but don't wait for processing
    criterion.add_to_buffer(torch.tensor(1.0))
    
    # Should fail if queue is not empty
    with pytest.raises(AssertionError, match="Buffer queue is not empty"):
        criterion.reset_buffer()


def test_error_resilience():
    """Test error resilience in buffer operations."""
    criterion = ConcreteCriterion()
    
    # Add normal data to test resilience
    criterion.add_to_buffer(torch.tensor(1.0))
    criterion._buffer_queue.join()
    
    # System should remain stable
    assert len(criterion.buffer) == 1
    assert criterion._buffer_thread.is_alive()