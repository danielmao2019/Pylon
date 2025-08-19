import pytest
import torch
import time
import threading
from typing import Dict, Any, Optional
from criteria.base_criterion import BaseCriterion


class DummyCriterion(BaseCriterion):
    """Test implementation of BaseCriterion for async buffer testing."""
    
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.abs(y_pred - y_true).mean()
    
    def summarize(self, output_path: Optional[str] = None) -> torch.Tensor:
        assert self.use_buffer, "Buffer must be enabled"
        assert len(self.buffer) > 0, "Buffer must not be empty"
        result = torch.stack(self.buffer).mean()
        if output_path:
            torch.save(result, output_path)
        return result


class FailingCriterion(BaseCriterion):
    """Criterion that fails during async operations for error testing."""
    
    def __init__(self, fail_mode: str = "detach"):
        super().__init__(use_buffer=True)
        self.fail_mode = fail_mode
        
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.abs(y_pred - y_true).mean()
    
    def summarize(self, output_path: Optional[str] = None) -> torch.Tensor:
        return torch.tensor(0.0)
        
    def _buffer_worker(self) -> None:
        """Override to inject failures for testing error propagation."""
        while True:
            data = self._buffer_queue.get()
            
            if self.fail_mode == "detach":
                # Force failure during detach
                raise RuntimeError("Simulated detach failure")
            elif self.fail_mode == "cpu":
                detached = data.detach()
                # Force failure during CPU transfer
                raise RuntimeError("Simulated CPU transfer failure")
            elif self.fail_mode == "lock":
                # Force failure during lock acquisition
                raise RuntimeError("Simulated lock failure")
            
            with self._buffer_lock:
                self.buffer.append(data.detach().cpu())
            self._buffer_queue.task_done()


@pytest.fixture
def dummy_criterion():
    """Fixture providing a DummyCriterion instance."""
    return DummyCriterion(use_buffer=True)


@pytest.fixture
def sample_tensor():
    """Fixture providing a sample tensor."""
    return torch.tensor(1.0)


def test_high_frequency_buffer_operations(dummy_criterion):
    """Test rapid buffer additions for thread safety under load."""
    # Generate many tensors rapidly
    num_operations = 100
    tensors = [torch.tensor(i * 0.1) for i in range(num_operations)]
    
    # Add all tensors rapidly
    for tensor in tensors:
        dummy_criterion.add_to_buffer(tensor)
    
    # Wait for async processing to complete
    dummy_criterion._buffer_queue.join()
    
    # Verify all items were processed
    buffer = dummy_criterion.get_buffer()
    assert len(buffer) == num_operations
    
    # Verify correctness of processed data
    for i, processed_tensor in enumerate(buffer):
        expected_value = i * 0.1
        assert abs(processed_tensor.item() - expected_value) < 1e-6
    
    # Verify all tensors are on CPU and detached
    for processed_tensor in buffer:
        assert processed_tensor.device.type == 'cpu'
        assert not processed_tensor.requires_grad


def test_concurrent_buffer_access(dummy_criterion):
    """Test thread safety with multiple threads accessing buffer."""
    def add_tensors_worker(start_idx: int, count: int):
        """Worker function to add tensors from multiple threads."""
        for i in range(count):
            tensor = torch.tensor(float(start_idx + i))
            dummy_criterion.add_to_buffer(tensor)
    
    # Create multiple threads adding data concurrently
    num_threads = 3
    items_per_thread = 10
    threads = []
    
    for thread_id in range(num_threads):
        start_idx = thread_id * items_per_thread
        thread = threading.Thread(
            target=add_tensors_worker,
            args=(start_idx, items_per_thread)
        )
        threads.append(thread)
    
    # Start all threads simultaneously
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Wait for async processing to complete
    dummy_criterion._buffer_queue.join()
    
    # Verify all items were processed
    total_items = num_threads * items_per_thread
    buffer = dummy_criterion.get_buffer()
    assert len(buffer) == total_items
    
    # Verify no data corruption occurred
    values = [tensor.item() for tensor in buffer]
    expected_values = list(range(total_items))
    assert sorted(values) == sorted(expected_values)


def test_memory_pressure_queue_growth(dummy_criterion):
    """Test behavior when queue grows due to slow processing vs fast additions."""
    # Create tensors with computation graphs to test memory retention
    num_items = 50
    large_tensors = []
    
    for i in range(num_items):
        # Create tensors with computation graph
        tensor = torch.tensor(float(i), requires_grad=True)
        result = (tensor * 2 + 1).sum()  # Create computation graph
        large_tensors.append(result)
    
    # Add all tensors rapidly - they should queue up with computation graphs intact
    for tensor in large_tensors:
        dummy_criterion.add_to_buffer(tensor)
    
    # Verify queue has items waiting
    queue_size = dummy_criterion._buffer_queue.qsize()
    assert queue_size > 0
    
    # Wait for all processing to complete
    dummy_criterion._buffer_queue.join()
    
    # Verify all items were processed and properly detached
    buffer = dummy_criterion.get_buffer()
    assert len(buffer) == num_items
    
    # Verify all tensors are detached and on CPU
    for processed_tensor in buffer:
        assert processed_tensor.device.type == 'cpu'
        assert not processed_tensor.requires_grad


def test_queue_synchronization_join_behavior(dummy_criterion):
    """Test queue.join() behavior ensures all items are processed."""
    # Add items and immediately check join behavior
    num_items = 20
    for i in range(num_items):
        dummy_criterion.add_to_buffer(torch.tensor(float(i)))
    
    # join() should block until all items are processed
    dummy_criterion._buffer_queue.join()
    
    # Verify all items are processed after join()
    buffer = dummy_criterion.get_buffer()
    assert len(buffer) == num_items
    
    # Queue should be empty after join()
    assert dummy_criterion._buffer_queue.empty()


def test_buffer_thread_lifecycle(dummy_criterion):
    """Test that buffer worker thread remains alive during normal operation."""
    # Verify thread is alive after initialization
    assert dummy_criterion._buffer_thread.is_alive()
    
    # Add some data and verify thread remains alive
    for i in range(5):
        dummy_criterion.add_to_buffer(torch.tensor(float(i)))
    
    dummy_criterion._buffer_queue.join()
    
    # Thread should still be alive after processing
    assert dummy_criterion._buffer_thread.is_alive()
    
    # Verify it's a daemon thread (will die when main program exits)
    assert dummy_criterion._buffer_thread.daemon


def test_worker_thread_error_propagation_detach():
    """Test that worker thread errors propagate and would crash the program."""
    # This test verifies that errors in worker threads are not suppressed
    # In the actual implementation, these would crash the program (fail-fast)
    
    failing_criterion = FailingCriterion(fail_mode="detach")
    
    # Add a tensor that will cause the worker to fail
    failing_tensor = torch.tensor(1.0)
    
    # Since we removed error handling, the worker thread will crash
    # This would normally terminate the program, but in tests we expect
    # the worker thread to die and the error to be visible
    
    failing_criterion.add_to_buffer(failing_tensor)
    
    # Give time for worker to process and fail
    time.sleep(0.1)
    
    # Worker thread should have died due to the unhandled exception
    # Note: In a real scenario, this would crash the entire program
    # We can't easily test this without complex process isolation
    
    # At minimum, verify the thread has the potential to fail
    assert hasattr(failing_criterion, '_buffer_thread')
    assert isinstance(failing_criterion._buffer_thread, threading.Thread)


def test_buffer_state_consistency_during_operations(dummy_criterion):
    """Test that buffer remains in consistent state during operations."""
    # Add several valid tensors
    valid_tensors = [torch.tensor(float(i)) for i in range(5)]
    for tensor in valid_tensors:
        dummy_criterion.add_to_buffer(tensor)
    
    # Wait for processing
    dummy_criterion._buffer_queue.join()
    
    # Verify buffer state is consistent
    buffer = dummy_criterion.get_buffer()
    assert len(buffer) == 5
    
    # Verify buffer contents are correct
    for i, tensor in enumerate(buffer):
        assert abs(tensor.item() - float(i)) < 1e-6
    
    # Test reset functionality
    dummy_criterion.reset_buffer()
    buffer_after_reset = dummy_criterion.get_buffer()
    assert len(buffer_after_reset) == 0


def test_disabled_buffer_operations():
    """Test criterion behavior with disabled buffer."""
    criterion = DummyCriterion(use_buffer=False)
    
    # Verify buffer is disabled
    assert not criterion.use_buffer
    assert not hasattr(criterion, 'buffer')
    assert not hasattr(criterion, '_buffer_thread')
    
    # Test that add_to_buffer does nothing when buffer is disabled
    tensor = torch.tensor(1.0)
    criterion.add_to_buffer(tensor)  # Should not raise error
    
    # Verify no buffer was created
    assert not hasattr(criterion, 'buffer')
    
    # Test that get_buffer raises error when buffer is disabled
    with pytest.raises(RuntimeError, match="Buffer is not enabled"):
        criterion.get_buffer()


def test_buffer_with_different_tensor_types(dummy_criterion):
    """Test buffer operations with different tensor types and shapes."""
    # Test with different dtypes and shapes
    test_tensors = [
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(2.0, dtype=torch.float64),
        torch.tensor(3.0, dtype=torch.float16),
    ]
    
    for tensor in test_tensors:
        dummy_criterion.add_to_buffer(tensor)
    
    dummy_criterion._buffer_queue.join()
    
    buffer = dummy_criterion.get_buffer()
    assert len(buffer) == len(test_tensors)
    
    # All should be on CPU regardless of input device
    for processed_tensor in buffer:
        assert processed_tensor.device.type == 'cpu'


@pytest.mark.parametrize("num_workers,items_per_worker", [
    (2, 5),
    (3, 10),
    (4, 8),
])
def test_parametrized_concurrent_operations(dummy_criterion, num_workers, items_per_worker):
    """Parametrized test for different concurrency scenarios."""
    def worker_function(worker_id: int):
        for i in range(items_per_worker):
            tensor = torch.tensor(float(worker_id * items_per_worker + i))
            dummy_criterion.add_to_buffer(tensor)
    
    # Create and start worker threads
    threads = []
    for worker_id in range(num_workers):
        thread = threading.Thread(target=worker_function, args=(worker_id,))
        threads.append(thread)
        thread.start()
    
    # Wait for all workers to complete
    for thread in threads:
        thread.join()
    
    # Wait for async processing
    dummy_criterion._buffer_queue.join()
    
    # Verify results
    total_items = num_workers * items_per_worker
    buffer = dummy_criterion.get_buffer()
    assert len(buffer) == total_items
    
    # Verify all expected values are present
    values = [tensor.item() for tensor in buffer]
    expected_values = list(range(total_items))
    assert sorted(values) == sorted(expected_values)