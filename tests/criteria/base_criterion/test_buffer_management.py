"""Test buffer management patterns for BaseCriterion."""
import time
import threading
import pytest
import torch
from unittest.mock import patch
from .conftest import ConcreteCriterion


def test_buffer_worker_thread_lifecycle():
    """Test buffer worker thread lifecycle."""
    criterion = ConcreteCriterion()
    
    # Thread should be alive after initialization
    assert criterion._buffer_thread.is_alive()
    assert criterion._buffer_thread.daemon is True
    
    # Add some data to test thread processing
    test_tensor = torch.tensor(1.0)
    criterion.add_to_buffer(test_tensor)
    
    # Wait for processing
    criterion._buffer_queue.join()
    
    # Verify data was processed
    assert len(criterion.buffer) == 1
    assert torch.equal(criterion.buffer[0], test_tensor.detach().cpu())


def test_concurrent_buffer_access():
    """Test concurrent access to buffer from multiple threads."""
    criterion = ConcreteCriterion()
    num_threads = 5
    items_per_thread = 10
    results = []
    
    def worker(thread_id):
        thread_results = []
        for i in range(items_per_thread):
            tensor = torch.tensor(float(thread_id * items_per_thread + i))
            criterion.add_to_buffer(tensor)
            thread_results.append(tensor.item())
        results.extend(thread_results)
    
    # Start multiple threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Wait for buffer processing
    criterion._buffer_queue.join()
    
    # Verify all items were processed
    assert len(criterion.buffer) == num_threads * items_per_thread
    buffer_values = [item.item() for item in criterion.buffer]
    assert set(buffer_values) == set(results)


def test_thread_safety_get_buffer():
    """Test thread-safe get_buffer operation."""
    criterion = ConcreteCriterion()
    
    # Add some data concurrently
    def add_data():
        for i in range(5):
            criterion.add_to_buffer(torch.tensor(float(i)))
            time.sleep(0.01)
    
    buffer_results = []
    
    def get_buffer_periodically():
        for _ in range(10):
            try:
                buffer_copy = criterion.get_buffer()
                buffer_results.append(len(buffer_copy))
            except Exception as e:
                buffer_results.append(f"Error: {e}")
            time.sleep(0.01)
    
    # Start threads
    add_thread = threading.Thread(target=add_data)
    get_thread = threading.Thread(target=get_buffer_periodically)
    
    add_thread.start()
    get_thread.start()
    
    add_thread.join()
    get_thread.join()
    
    # Wait for final processing
    criterion._buffer_queue.join()
    
    # Should not have any errors
    errors = [r for r in buffer_results if isinstance(r, str)]
    assert len(errors) == 0


def test_buffer_worker_error_handling():
    """Test buffer worker error handling."""
    criterion = ConcreteCriterion()
    
    # Add valid data first to ensure thread is working
    criterion.add_to_buffer(torch.tensor(1.0))
    criterion._buffer_queue.join()
    
    # Thread should still be alive after processing
    assert criterion._buffer_thread.is_alive()
    
    # The worker thread handles errors internally and prints them
    # but doesn't crash the thread - this is the expected behavior


def test_add_to_buffer_valid_tensor():
    """Test adding valid tensors to buffer."""
    criterion = ConcreteCriterion()
    
    # Test scalar tensor
    scalar_tensor = torch.tensor(1.5)
    criterion.add_to_buffer(scalar_tensor)
    criterion._buffer_queue.join()
    
    assert len(criterion.buffer) == 1
    assert torch.equal(criterion.buffer[0], scalar_tensor.detach().cpu())


@pytest.mark.parametrize("invalid_tensor", [
    torch.tensor([1.0, 2.0]),  # Non-scalar tensor
    torch.tensor([[1.0]]),     # Multi-dimensional tensor  
    1.0,                       # Non-tensor
])
def test_add_to_buffer_invalid_tensor(invalid_tensor):
    """Test adding invalid tensors to buffer."""
    criterion = ConcreteCriterion()
    
    with pytest.raises(AssertionError):
        criterion.add_to_buffer(invalid_tensor)


@pytest.mark.parametrize("invalid_value,check_validity", [
    (float('nan'), True),
    (float('inf'), True),
    (float('-inf'), True),
])
def test_add_to_buffer_invalid_values_with_check(invalid_value, check_validity):
    """Test adding NaN and infinite values with validity checking."""
    criterion = ConcreteCriterion()
    
    with pytest.raises(AssertionError):
        criterion.add_to_buffer(torch.tensor(invalid_value))


def test_add_to_buffer_invalid_values_without_check():
    """Test adding NaN and infinite values without validity checking."""
    criterion = ConcreteCriterion()
    
    # Should work with check_validity=False
    criterion.add_to_buffer(torch.tensor(float('nan')), check_validity=False)
    criterion.add_to_buffer(torch.tensor(float('inf')), check_validity=False)
    criterion._buffer_queue.join()
    
    assert len(criterion.buffer) == 2


def test_get_buffer_thread_safe():
    """Test thread-safe buffer retrieval."""
    criterion = ConcreteCriterion()
    
    # Add some data
    for i in range(5):
        criterion.add_to_buffer(torch.tensor(float(i)))
    
    criterion._buffer_queue.join()
    
    # Get buffer should return a copy
    buffer_copy = criterion.get_buffer()
    assert len(buffer_copy) == 5
    assert buffer_copy is not criterion.buffer
    
    # Modifying copy should not affect original
    buffer_copy.append(torch.tensor(999.0))
    assert len(criterion.buffer) == 5


def test_get_buffer_without_buffer_enabled():
    """Test get_buffer when buffer is disabled."""
    criterion = ConcreteCriterion(use_buffer=False)
    
    with pytest.raises(RuntimeError, match="Buffer is not enabled"):
        criterion.get_buffer()


def test_reset_buffer_with_buffer_enabled():
    """Test resetting buffer when enabled."""
    criterion = ConcreteCriterion()
    
    # Add some data
    for i in range(3):
        criterion.add_to_buffer(torch.tensor(float(i)))
    
    criterion._buffer_queue.join()
    assert len(criterion.buffer) == 3
    
    # Reset buffer
    criterion.reset_buffer()
    assert len(criterion.buffer) == 0
    assert isinstance(criterion.buffer, list)


def test_reset_buffer_without_buffer_enabled():
    """Test reset buffer when buffer is disabled."""
    criterion = ConcreteCriterion(use_buffer=False)
    
    # Reset should work without errors when buffer is disabled
    # It just checks that buffer attribute doesn't exist
    criterion.reset_buffer()
    assert not hasattr(criterion, 'buffer')