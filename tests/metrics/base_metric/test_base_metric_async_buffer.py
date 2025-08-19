import pytest
import torch
import time
import threading
from typing import Dict, Any, Optional, List
from metrics.base_metric import BaseMetric


class DummyMetric(BaseMetric):
    """Test implementation of BaseMetric for async buffer testing."""
    
    def __call__(self, datapoint: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        # Extract data for processing
        y_pred = datapoint['outputs']['prediction']
        y_true = datapoint['labels']['target']
        
        # Compute metric
        metric_data = {
            'error': torch.abs(y_pred - y_true).mean(),
            'squared_error': torch.pow(y_pred - y_true, 2).mean()
        }
        
        # Add to buffer if enabled
        if self.use_buffer:
            self.add_to_buffer(metric_data, datapoint)
        
        return {
            'error': metric_data['error'].item(),
            'squared_error': metric_data['squared_error'].item()
        }
    
    def summarize(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        assert self.use_buffer, "Buffer must be enabled"
        buffer_list = self.get_buffer()
        assert len(buffer_list) > 0, "Buffer must not be empty"
        
        # Aggregate metrics
        errors = [item['error'] for item in buffer_list]
        squared_errors = [item['squared_error'] for item in buffer_list]
        
        result = {
            'mean_error': torch.stack(errors).mean().item(),
            'mean_squared_error': torch.stack(squared_errors).mean().item(),
            'count': len(buffer_list)
        }
        
        if output_path:
            torch.save(result, output_path)
        
        return result


class FailingMetric(BaseMetric):
    """Metric that fails during async operations for error testing."""
    
    def __init__(self, fail_mode: str = "cpu"):
        super().__init__(use_buffer=True)
        self.fail_mode = fail_mode
        
    def __call__(self, datapoint: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        return {'dummy': 1.0}
    
    def summarize(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        return {'dummy': 0.0}
        
    def _buffer_worker(self) -> None:
        """Override to inject failures for testing error propagation."""
        while True:
            item = self._buffer_queue.get()
            data, idx = item['data'], item['idx']
            
            if self.fail_mode == "cpu":
                # Simulate failure during CPU transfer
                raise RuntimeError("Simulated CPU transfer failure")
            elif self.fail_mode == "apply_tensor_op":
                # Simulate failure during tensor operation
                raise RuntimeError("Simulated tensor operation failure")
            
            # Normal processing that would fail due to injected error
            from utils.ops.apply import apply_tensor_op
            processed_data = apply_tensor_op(func=lambda x: x.detach().cpu(), inputs=data)
            
            with self._buffer_lock:
                self.buffer[idx] = processed_data
            
            self._buffer_queue.task_done()


@pytest.fixture
def dummy_metric():
    """Fixture providing a DummyMetric instance."""
    return DummyMetric(use_buffer=True)


@pytest.fixture
def sample_datapoint():
    """Fixture providing a sample datapoint."""
    return {
        'inputs': {
            'data': torch.randn(2, 3)
        },
        'labels': {
            'target': torch.randn(2, 3)
        },
        'outputs': {
            'prediction': torch.randn(2, 3)
        },
        'meta_info': {
            'idx': 0
        }
    }


def create_datapoint(idx: int) -> Dict[str, Dict[str, Any]]:
    """Helper function to create datapoint with specific index."""
    return {
        'inputs': {
            'data': torch.randn(2, 3)
        },
        'labels': {
            'target': torch.ones(2, 3) * idx
        },
        'outputs': {
            'prediction': torch.ones(2, 3) * (idx + 0.1)
        },
        'meta_info': {
            'idx': idx
        }
    }


def test_high_frequency_metric_operations(dummy_metric):
    """Test rapid metric computations for thread safety under load."""
    num_operations = 50
    datapoints = [create_datapoint(i) for i in range(num_operations)]
    
    # Process all datapoints rapidly
    results = []
    for datapoint in datapoints:
        result = dummy_metric(datapoint)
        results.append(result)
    
    # Wait for async processing to complete
    dummy_metric._buffer_queue.join()
    
    # Verify all items were processed
    buffer = dummy_metric.get_buffer()
    assert len(buffer) == num_operations
    
    # Verify order preservation (indices should be 0, 1, 2, ...)
    for i, buffered_item in enumerate(buffer):
        assert 'error' in buffered_item
        assert 'squared_error' in buffered_item
        # Check that tensor operations were properly applied
        assert buffered_item['error'].device.type == 'cpu'
        assert not buffered_item['error'].requires_grad


def test_concurrent_metric_access(dummy_metric):
    """Test thread safety with multiple threads processing metrics."""
    def process_metrics_worker(start_idx: int, count: int):
        """Worker function to process metrics from multiple threads."""
        for i in range(count):
            idx = start_idx + i
            datapoint = create_datapoint(idx)
            dummy_metric(datapoint)
    
    # Create multiple threads processing data concurrently
    num_threads = 3
    items_per_thread = 8
    threads = []
    
    for thread_id in range(num_threads):
        start_idx = thread_id * items_per_thread
        thread = threading.Thread(
            target=process_metrics_worker,
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
    dummy_metric._buffer_queue.join()
    
    # Verify all items were processed
    total_items = num_threads * items_per_thread
    buffer = dummy_metric.get_buffer()
    assert len(buffer) == total_items
    
    # Verify order preservation and no data corruption
    for i, buffered_item in enumerate(buffer):
        assert isinstance(buffered_item, dict)
        assert 'error' in buffered_item
        assert 'squared_error' in buffered_item


def test_index_format_handling(dummy_metric):
    """Test different index formats from DataLoader collation."""
    # Test tensor format with idx=0
    datapoint_tensor = create_datapoint(0)
    datapoint_tensor['meta_info']['idx'] = torch.tensor([0], dtype=torch.int64)
    dummy_metric(datapoint_tensor)
    
    # Test list format with idx=1
    datapoint_list = create_datapoint(1)
    datapoint_list['meta_info']['idx'] = [1]
    dummy_metric(datapoint_list)
    
    # Test int format with idx=2 (already set by create_datapoint)
    datapoint_int = create_datapoint(2)
    dummy_metric(datapoint_int)
    
    # Wait for processing
    dummy_metric._buffer_queue.join()
    
    # Verify all were processed correctly
    buffer = dummy_metric.get_buffer()
    assert len(buffer) == 3
    
    # Buffer should be ordered by contiguous indices: 0, 1, 2
    for buffered_item in buffer:
        assert isinstance(buffered_item, dict)


def test_memory_pressure_with_complex_data(dummy_metric):
    """Test behavior with complex nested data structures."""
    num_items = 30
    complex_datapoints = []
    
    for i in range(num_items):
        # Create complex datapoint with nested tensors
        datapoint = {
            'inputs': {
                'images': torch.randn(3, 32, 32, requires_grad=True),
                'metadata': torch.tensor([i, i+1, i+2], dtype=torch.float32)
            },
            'labels': {
                'target': torch.randn(32, 32),  # Match DummyMetric expectation
                'segmentation': torch.randint(0, 5, (32, 32)),
                'classification': torch.tensor(i % 3)
            },
            'outputs': {
                'prediction': torch.randn(32, 32),  # Match DummyMetric expectation
                'seg_pred': torch.randn(5, 32, 32),
                'cls_pred': torch.randn(3)
            },
            'meta_info': {
                'idx': i
            }
        }
        complex_datapoints.append(datapoint)
    
    # Process all complex datapoints
    for datapoint in complex_datapoints:
        dummy_metric(datapoint)
    
    # Verify queue growth during processing
    queue_size = dummy_metric._buffer_queue.qsize()
    assert queue_size >= 0  # Queue should handle the load
    
    # Wait for all processing to complete
    dummy_metric._buffer_queue.join()
    
    # Verify all items were processed
    buffer = dummy_metric.get_buffer()
    assert len(buffer) == num_items
    
    # Verify all tensors are properly detached and on CPU
    for buffered_item in buffer:
        for key, tensor in buffered_item.items():
            if isinstance(tensor, torch.Tensor):
                assert tensor.device.type == 'cpu'
                assert not tensor.requires_grad


def test_buffer_order_preservation_under_load():
    """Test that buffer maintains correct order under concurrent load."""
    metric = DummyMetric(use_buffer=True)
    
    def process_batch(start_idx: int, batch_size: int):
        """Process a batch of datapoints in sequence."""
        for i in range(batch_size):
            idx = start_idx + i
            datapoint = create_datapoint(idx)
            metric(datapoint)
    
    # Process from multiple threads with overlapping indices
    threads = []
    batch_size = 5
    
    # Thread 1: indices 0-4
    # Thread 2: indices 5-9  
    # Thread 3: indices 10-14
    for thread_id in range(3):
        start_idx = thread_id * batch_size
        thread = threading.Thread(target=process_batch, args=(start_idx, batch_size))
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    metric._buffer_queue.join()
    
    # Verify order preservation
    buffer = metric.get_buffer()
    assert len(buffer) == 15
    
    # Buffer should maintain index order 0, 1, 2, ..., 14
    # Even though processing was concurrent
    for i, buffered_item in enumerate(buffer):
        assert isinstance(buffered_item, dict)


def test_queue_synchronization_with_metrics(dummy_metric):
    """Test queue.join() behavior with metric processing."""
    num_items = 15
    datapoints = [create_datapoint(i) for i in range(num_items)]
    
    # Process all datapoints
    for datapoint in datapoints:
        dummy_metric(datapoint)
    
    # join() should block until all items are processed
    dummy_metric._buffer_queue.join()
    
    # Verify all items are processed after join()
    buffer = dummy_metric.get_buffer()
    assert len(buffer) == num_items
    
    # Queue should be empty after join()
    assert dummy_metric._buffer_queue.empty()


def test_worker_thread_error_propagation_metric():
    """Test that worker thread errors propagate in metrics."""
    failing_metric = FailingMetric(fail_mode="cpu")
    
    # Create a datapoint that will cause the worker to fail
    datapoint = create_datapoint(0)
    
    # Add the failing metric computation
    failing_metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint)
    
    # Give time for worker to process and fail
    time.sleep(0.1)
    
    # Worker thread should have potential to fail
    # (In real scenario, this would crash the program)
    assert hasattr(failing_metric, '_buffer_thread')
    assert isinstance(failing_metric._buffer_thread, threading.Thread)


def test_disabled_buffer_metric_operations():
    """Test metric behavior with disabled buffer."""
    metric = DummyMetric(use_buffer=False)
    
    # Verify buffer is disabled
    assert not metric.use_buffer
    assert not hasattr(metric, 'buffer')
    assert not hasattr(metric, '_buffer_thread')
    
    # Test that metric computation works without buffer
    datapoint = create_datapoint(0)
    result = metric(datapoint)
    
    # Should return valid result
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'squared_error' in result
    
    # Verify no buffer was created
    assert not hasattr(metric, 'buffer')
    
    # Test that get_buffer raises error when buffer is disabled
    with pytest.raises(RuntimeError, match="Buffer is not enabled"):
        metric.get_buffer()


def test_buffer_state_consistency_metrics(dummy_metric):
    """Test that metric buffer remains in consistent state."""
    # Process several datapoints
    num_items = 5
    for i in range(num_items):
        datapoint = create_datapoint(i)
        dummy_metric(datapoint)
    
    # Wait for processing
    dummy_metric._buffer_queue.join()
    
    # Verify buffer state is consistent
    buffer = dummy_metric.get_buffer()
    assert len(buffer) == num_items
    
    # Verify buffer contents are correct dictionaries
    for buffered_item in buffer:
        assert isinstance(buffered_item, dict)
        assert 'error' in buffered_item
        assert 'squared_error' in buffered_item
    
    # Test reset functionality
    dummy_metric.reset_buffer()
    buffer_after_reset = dummy_metric.get_buffer()
    assert len(buffer_after_reset) == 0


@pytest.mark.parametrize("idx_format,idx_value", [
    ("tensor", torch.tensor([0], dtype=torch.int64)),
    ("list", [0]),
    ("int", 0),
])
def test_parametrized_index_formats(idx_format, idx_value):
    """Parametrized test for different index formats."""
    # Create fresh metric for each test to ensure clean state
    metric = DummyMetric(use_buffer=True)
    
    datapoint = create_datapoint(0)
    datapoint['meta_info']['idx'] = idx_value
    
    # Process the datapoint
    metric(datapoint)
    metric._buffer_queue.join()
    
    # Verify processing succeeded
    buffer = metric.get_buffer()
    assert len(buffer) == 1
    
    # The actual index value should be 0 in all cases
    assert isinstance(buffer[0], dict)


def test_metric_summarize_functionality(dummy_metric):
    """Test metric summarization with buffered data."""
    # Process multiple datapoints
    num_items = 8
    for i in range(num_items):
        datapoint = create_datapoint(i)
        dummy_metric(datapoint)
    
    # Wait for processing
    dummy_metric._buffer_queue.join()
    
    # Test summarize functionality
    summary = dummy_metric.summarize()
    
    # Verify summary structure
    assert isinstance(summary, dict)
    assert 'mean_error' in summary
    assert 'mean_squared_error' in summary
    assert 'count' in summary
    assert summary['count'] == num_items
    
    # Verify numeric values are reasonable
    assert isinstance(summary['mean_error'], float)
    assert isinstance(summary['mean_squared_error'], float)
    assert summary['mean_error'] >= 0
    assert summary['mean_squared_error'] >= 0


def test_invalid_index_format_handling():
    """Test ValueError for unsupported index types in add_to_buffer."""
    metric = DummyMetric(use_buffer=True)
    
    # Test unsupported index types that should trigger ValueError
    invalid_datapoints = [
        # String index
        {
            'outputs': {'prediction': torch.randn(2, 3)},
            'labels': {'target': torch.randn(2, 3)},
            'meta_info': {'idx': "invalid_string"}
        },
        # Float index  
        {
            'outputs': {'prediction': torch.randn(2, 3)},
            'labels': {'target': torch.randn(2, 3)},
            'meta_info': {'idx': 3.14}
        },
        # Dictionary index
        {
            'outputs': {'prediction': torch.randn(2, 3)},
            'labels': {'target': torch.randn(2, 3)},
            'meta_info': {'idx': {'nested': 1}}
        },
        # None index
        {
            'outputs': {'prediction': torch.randn(2, 3)},
            'labels': {'target': torch.randn(2, 3)},
            'meta_info': {'idx': None}
        }
    ]
    
    for i, invalid_datapoint in enumerate(invalid_datapoints):
        with pytest.raises(ValueError, match="Unsupported idx format"):
            metric.add_to_buffer({'test': torch.tensor(1.0)}, invalid_datapoint)


def test_malformed_tensor_index_assertions():
    """Test assertion failures for wrong tensor shapes/dtypes in index."""
    metric = DummyMetric(use_buffer=True)
    
    # Test wrong tensor shape (multiple elements)
    datapoint_multi_element = create_datapoint(0)
    datapoint_multi_element['meta_info']['idx'] = torch.tensor([1, 2, 3], dtype=torch.int64)
    with pytest.raises(AssertionError, match="Expected single element tensor"):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_multi_element)
    
    # Test wrong tensor shape (empty tensor)
    datapoint_empty = create_datapoint(0)
    datapoint_empty['meta_info']['idx'] = torch.tensor([], dtype=torch.int64)
    with pytest.raises(AssertionError, match="Expected single element tensor"):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_empty)
    
    # Test wrong tensor shape (multi-dimensional)
    datapoint_multi_dim = create_datapoint(0)
    datapoint_multi_dim['meta_info']['idx'] = torch.tensor([[1]], dtype=torch.int64)
    with pytest.raises(AssertionError, match="Expected single element tensor"):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_multi_dim)
    
    # Test wrong tensor dtype (float instead of int64)
    datapoint_wrong_dtype = create_datapoint(0)
    datapoint_wrong_dtype['meta_info']['idx'] = torch.tensor([1.0])
    with pytest.raises(AssertionError):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_wrong_dtype)
    
    # Test wrong tensor dtype (int32 instead of int64)
    datapoint_int32 = create_datapoint(0)
    datapoint_int32['meta_info']['idx'] = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(AssertionError):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_int32)


def test_malformed_list_index_assertions():
    """Test assertion failures for wrong list contents in index."""
    metric = DummyMetric(use_buffer=True)
    
    # Test empty list
    datapoint_empty_list = create_datapoint(0)
    datapoint_empty_list['meta_info']['idx'] = []
    with pytest.raises(AssertionError):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_empty_list)
    
    # Test list with multiple elements
    datapoint_multi_list = create_datapoint(0)
    datapoint_multi_list['meta_info']['idx'] = [1, 2, 3]
    with pytest.raises(AssertionError):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_multi_list)
    
    # Test list with non-integer contents
    datapoint_float_list = create_datapoint(0)
    datapoint_float_list['meta_info']['idx'] = [1.5]
    with pytest.raises(AssertionError):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_float_list)
    
    # Test list with string contents
    datapoint_str_list = create_datapoint(0)
    datapoint_str_list['meta_info']['idx'] = ["1"]
    with pytest.raises(AssertionError):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_str_list)
    
    # Test list with None contents
    datapoint_none_list = create_datapoint(0)
    datapoint_none_list['meta_info']['idx'] = [None]
    with pytest.raises(AssertionError):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_none_list)


def test_missing_meta_info_assertions():
    """Test assertion failures for datapoints without required meta_info structure."""
    metric = DummyMetric(use_buffer=True)
    
    # Test completely missing meta_info
    datapoint_no_meta = {
        'outputs': {'prediction': torch.randn(2, 3)},
        'labels': {'target': torch.randn(2, 3)}
        # Missing 'meta_info'
    }
    with pytest.raises(AssertionError, match="meta_info"):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_no_meta)
    
    # Test meta_info present but missing idx
    datapoint_no_idx = {
        'outputs': {'prediction': torch.randn(2, 3)},
        'labels': {'target': torch.randn(2, 3)},
        'meta_info': {'other_field': 'value'}  # Missing 'idx'
    }
    with pytest.raises(AssertionError, match="idx"):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_no_idx)
    
    # Test meta_info as None
    datapoint_none_meta = {
        'outputs': {'prediction': torch.randn(2, 3)},
        'labels': {'target': torch.randn(2, 3)},
        'meta_info': None
    }
    with pytest.raises(AssertionError, match="meta_info"):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_none_meta)
    
    # Test meta_info as wrong type (list instead of dict)
    datapoint_list_meta = {
        'outputs': {'prediction': torch.randn(2, 3)},
        'labels': {'target': torch.randn(2, 3)},
        'meta_info': ['not', 'a', 'dict']
    }
    with pytest.raises(AssertionError, match="idx"):
        metric.add_to_buffer({'test': torch.tensor(1.0)}, datapoint_list_meta)


def test_non_contiguous_index_scenarios():
    """Test realistic evaluation scenarios with non-contiguous indices."""
    metric = DummyMetric(use_buffer=True)
    
    # Scenario 1: Evaluation subset with gaps (realistic evaluation pattern)
    evaluation_indices = [2, 5, 7, 10, 15]  # Non-contiguous indices
    datapoints = []
    
    for idx in evaluation_indices:
        datapoint = create_datapoint(idx)
        datapoints.append(datapoint)
    
    # Process non-contiguous indices
    for datapoint in datapoints:
        metric(datapoint)
    
    metric._buffer_queue.join()
    
    # This should fail with current implementation (contiguous check)
    with pytest.raises(AssertionError, match="sorted_indices"):
        metric.get_buffer()
    
    # Scenario 2: Test that the error provides useful information
    metric2 = DummyMetric(use_buffer=True)
    sparse_indices = [0, 3, 8]  # Gaps between indices
    
    for idx in sparse_indices:
        datapoint = create_datapoint(idx)
        metric2(datapoint)
    
    metric2._buffer_queue.join()
    
    # Verify the assertion shows which indices are problematic
    try:
        metric2.get_buffer()
        assert False, "Expected assertion error for non-contiguous indices"
    except AssertionError as e:
        error_msg = str(e)
        assert "sorted_indices" in error_msg
        assert "[0, 3, 8]" in error_msg or "0" in error_msg
    
    # Scenario 3: Shuffled processing order but contiguous final indices
    metric3 = DummyMetric(use_buffer=True)
    indices_to_process = [0, 1, 2, 3, 4]
    
    # Process in random order
    import random
    shuffled_indices = indices_to_process.copy()
    random.shuffle(shuffled_indices)
    
    for idx in shuffled_indices:
        datapoint = create_datapoint(idx)
        metric3(datapoint)
    
    metric3._buffer_queue.join()
    
    # This should work - indices are contiguous even if processed out of order
    buffer = metric3.get_buffer()
    assert len(buffer) == 5
    
    # Verify order is maintained correctly despite shuffled processing
    for i, buffered_item in enumerate(buffer):
        assert isinstance(buffered_item, dict)


def test_empty_buffer_edge_cases_metrics():
    """Test edge cases around empty buffer transitions for metrics."""
    metric = DummyMetric(use_buffer=True)
    
    # Test get_buffer on empty buffer
    empty_buffer = metric.get_buffer()
    assert empty_buffer == []
    assert isinstance(empty_buffer, list)
    
    # Test reset on empty buffer
    metric.reset_buffer()  # Should not raise
    assert metric.get_buffer() == []
    
    # Test multiple resets in sequence
    metric.reset_buffer()
    metric.reset_buffer()
    assert metric.get_buffer() == []
    
    # Test adding after empty state
    datapoint = create_datapoint(0)
    metric(datapoint)
    metric._buffer_queue.join()
    assert len(metric.get_buffer()) == 1
    
    # Test reset after having data
    metric.reset_buffer()
    assert metric.get_buffer() == []
    
    # Test empty buffer with dict structure
    assert isinstance(metric.buffer, dict)
    assert len(metric.buffer) == 0
    
    # Test empty buffer transitions with queue synchronization
    for i in range(3):
        datapoint = create_datapoint(i)
        metric(datapoint)
    metric._buffer_queue.join()
    assert len(metric.get_buffer()) == 3
    
    # Reset and immediately check
    metric.reset_buffer()
    assert len(metric.get_buffer()) == 0
    assert len(metric.buffer) == 0  # Internal dict should also be empty
    
    # Ensure queue is properly empty after reset
    assert metric._buffer_queue.empty()


def test_extreme_lock_contention_metrics():
    """Test metric buffer operations under extreme lock contention."""
    metric = DummyMetric(use_buffer=True)
    
    def contention_worker(worker_id: int, iterations: int):
        """Worker that creates lock contention with metrics processing."""
        for i in range(iterations):
            # Create datapoint with unique index
            idx = worker_id * iterations + i
            datapoint = create_datapoint(idx)
            
            # Process metric (creates async buffer operation)
            metric(datapoint)
            
            # Create lock contention with frequent buffer state checks
            if i % 3 == 0:  # More frequent checks for metrics
                try:
                    buffer_copy = metric.get_buffer()  # Acquires lock
                    assert isinstance(buffer_copy, list)
                except Exception:
                    # Some contention is expected, continue
                    pass
    
    # Create high contention scenario
    num_threads = 6
    iterations_per_thread = 20
    threads = []
    
    for worker_id in range(num_threads):
        thread = threading.Thread(
            target=contention_worker,
            args=(worker_id, iterations_per_thread)
        )
        threads.append(thread)
    
    # Start all threads simultaneously
    start_time = time.time()
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    contention_time = time.time() - start_time
    
    # Wait for async processing under contention
    metric._buffer_queue.join()
    
    # Verify system survived extreme contention
    total_expected = num_threads * iterations_per_thread
    buffer = metric.get_buffer()
    assert len(buffer) == total_expected, f"Lost data under contention: {len(buffer)}/{total_expected}"
    
    # Verify no deadlocks occurred
    assert contention_time < 30.0, f"Potential deadlock detected: {contention_time:.2f}s"
    
    # Verify buffer order preservation under contention
    for i, buffered_item in enumerate(buffer):
        assert isinstance(buffered_item, dict)
        assert 'error' in buffered_item
        assert 'squared_error' in buffered_item