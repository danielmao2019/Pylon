import pytest
import torch
import time
import threading


def test_buffer_worker_threading_setup(sequential_debugger_basic):
    """Test that buffer worker thread is properly set up."""
    debugger = sequential_debugger_basic
    
    # Test threading attributes exist
    assert hasattr(debugger, '_buffer_thread')
    assert hasattr(debugger, '_buffer_queue')
    assert hasattr(debugger, '_buffer_lock')
    
    # Test thread is running and configured correctly
    assert debugger._buffer_thread.is_alive()
    assert debugger._buffer_thread.daemon  # Should be daemon thread
    assert hasattr(debugger._buffer_lock, 'acquire')  # Check it's a lock-like object
    assert hasattr(debugger._buffer_lock, 'release')


def test_buffer_add_and_processing(sequential_debugger_basic, sample_datapoint):
    """Test basic buffer add functionality and async processing."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    # Create debug outputs
    debug_outputs = {
        'test_debugger': {
            'tensor_data': torch.randn(3, 32, 32),
            'stats': {'mean': 0.5, 'std': 1.0}
        }
    }
    
    # Add to buffer
    debugger.add_to_buffer(debug_outputs, sample_datapoint)
    
    # Wait for background processing
    debugger._buffer_queue.join()
    
    # Check that data was processed and added to current page
    with debugger._buffer_lock:
        assert len(debugger.current_page_data) == 1
        assert 0 in debugger.current_page_data  # idx from sample_datapoint
        
        # Check data was moved to CPU by apply_tensor_op
        stored_data = debugger.current_page_data[0]
        assert stored_data['test_debugger']['tensor_data'].device.type == 'cpu'


@pytest.mark.parametrize("idx_format,expected_idx", [
    ([0], 0),  # List format
    (torch.tensor([1], dtype=torch.int64), 1),  # Tensor format  
    (2, 2),  # Direct int format
])
def test_buffer_idx_format_handling(sequential_debugger_basic, idx_format, expected_idx):
    """Test buffer handles different idx formats correctly."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    # Create datapoint with specific idx format
    datapoint = {
        'inputs': torch.randn(1, 3, 32, 32, dtype=torch.float32),
        'outputs': torch.randn(1, 10, dtype=torch.float32),
        'meta_info': {'idx': idx_format}
    }
    
    debug_outputs = {'test': {'data': torch.randn(3, 3)}}
    
    # Add to buffer
    debugger.add_to_buffer(debug_outputs, datapoint)
    debugger._buffer_queue.join()
    
    # Check correct idx was extracted
    with debugger._buffer_lock:
        assert expected_idx in debugger.current_page_data


def test_buffer_multiple_datapoints(sequential_debugger_basic, multiple_datapoints):
    """Test buffer handles multiple datapoints correctly."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    # Add multiple debug outputs
    for i, datapoint in enumerate(multiple_datapoints):
        debug_outputs = {
            'test_debugger': {
                'datapoint_id': i,
                'tensor': torch.randn(10, 10),
                'metadata': {'processed_at': time.time()}
            }
        }
        debugger.add_to_buffer(debug_outputs, datapoint)
    
    # Wait for all processing to complete
    debugger._buffer_queue.join()
    
    # Check all datapoints were processed
    with debugger._buffer_lock:
        assert len(debugger.current_page_data) == len(multiple_datapoints)
        
        # Check indices are correct (0, 1, 2, 3, 4)
        expected_indices = set(range(len(multiple_datapoints)))
        actual_indices = set(debugger.current_page_data.keys())
        assert actual_indices == expected_indices
        
        # Check each datapoint has correct ID
        for i in range(len(multiple_datapoints)):
            assert debugger.current_page_data[i]['test_debugger']['datapoint_id'] == i


def test_buffer_disabled_state(sequential_debugger_basic, sample_datapoint):
    """Test buffer doesn't process when debugger is disabled."""
    debugger = sequential_debugger_basic
    debugger.enabled = False  # Disable debugger
    
    debug_outputs = {'test': {'data': torch.randn(5, 5)}}
    
    # Try to add to buffer when disabled
    debugger.add_to_buffer(debug_outputs, sample_datapoint)
    debugger._buffer_queue.join()
    
    # Should not have added anything
    with debugger._buffer_lock:
        assert len(debugger.current_page_data) == 0


def test_buffer_memory_size_calculation(sequential_debugger_basic, sample_datapoint):
    """Test that buffer calculates memory size correctly."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    # Create debug outputs with known size
    large_tensor = torch.randn(100, 100)  # Relatively large tensor
    debug_outputs = {
        'test_debugger': {
            'large_tensor': large_tensor,
            'small_data': {'count': 1}
        }
    }
    
    initial_page_size = debugger.current_page_size
    
    # Add to buffer
    debugger.add_to_buffer(debug_outputs, sample_datapoint)
    debugger._buffer_queue.join()
    
    # Check that page size increased
    with debugger._buffer_lock:
        assert debugger.current_page_size > initial_page_size
        assert debugger.current_page_size > 0


def test_buffer_apply_tensor_op_cpu_conversion(sequential_debugger_basic, sample_datapoint):
    """Test that apply_tensor_op correctly moves tensors to CPU."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    # Create debug outputs with GPU tensors (if available)
    if torch.cuda.is_available():
        gpu_tensor = torch.randn(5, 5).cuda()
        debug_outputs = {
            'test_debugger': {
                'gpu_tensor': gpu_tensor,
                'cpu_tensor': torch.randn(3, 3),  # Already on CPU
                'nested': {'inner_gpu': torch.randn(2, 2).cuda()}
            }
        }
        
        # Verify tensors start on GPU
        assert debug_outputs['test_debugger']['gpu_tensor'].device.type == 'cuda'
        assert debug_outputs['test_debugger']['nested']['inner_gpu'].device.type == 'cuda'
    else:
        # Fallback for CPU-only testing
        debug_outputs = {
            'test_debugger': {
                'cpu_tensor1': torch.randn(5, 5),
                'cpu_tensor2': torch.randn(3, 3),
                'nested': {'inner_cpu': torch.randn(2, 2)}
            }
        }
    
    # Add to buffer
    debugger.add_to_buffer(debug_outputs, sample_datapoint)
    debugger._buffer_queue.join()
    
    # Check all tensors are on CPU after processing
    with debugger._buffer_lock:
        stored_data = debugger.current_page_data[0]['test_debugger']
        
        # Check all tensors moved to CPU
        for key, value in stored_data.items():
            if isinstance(value, torch.Tensor):
                assert value.device.type == 'cpu', f"Tensor {key} not moved to CPU"
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, torch.Tensor):
                        assert nested_value.device.type == 'cpu', f"Nested tensor {nested_key} not moved to CPU"


def test_buffer_thread_safety_concurrent_access(sequential_debugger_basic):
    """Test buffer thread safety with concurrent access."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    # Create multiple threads that add to buffer simultaneously
    num_threads = 5
    items_per_thread = 10
    results = []
    
    def add_items(thread_id):
        """Function to add items from a specific thread."""
        for i in range(items_per_thread):
            datapoint = {
                'inputs': torch.randn(1, 3, 32, 32, dtype=torch.float32),
                'outputs': torch.randn(1, 10, dtype=torch.float32),
                'meta_info': {'idx': thread_id * items_per_thread + i}
            }
            debug_outputs = {
                'test_debugger': {
                    'thread_id': thread_id,
                    'item_id': i,
                    'data': torch.randn(3, 3)
                }
            }
            debugger.add_to_buffer(debug_outputs, datapoint)
        results.append(f"Thread {thread_id} completed")
    
    # Start multiple threads
    threads = []
    for thread_id in range(num_threads):
        thread = threading.Thread(target=add_items, args=(thread_id,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Wait for buffer processing to complete
    debugger._buffer_queue.join()
    
    # Check all items were processed correctly
    with debugger._buffer_lock:
        expected_total_items = num_threads * items_per_thread
        assert len(debugger.current_page_data) == expected_total_items
        
        # Check no data corruption occurred
        for idx, data in debugger.current_page_data.items():
            assert 'test_debugger' in data
            assert 'thread_id' in data['test_debugger']
            assert 'item_id' in data['test_debugger']
            assert isinstance(data['test_debugger']['data'], torch.Tensor)
    
    # Check all threads completed
    assert len(results) == num_threads


def test_buffer_error_handling(sequential_debugger_basic):
    """Test buffer worker handles errors gracefully."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    # Test with malformed datapoint (missing meta_info)
    malformed_datapoint = {
        'inputs': torch.randn(1, 3, 32, 32, dtype=torch.float32),
        'outputs': torch.randn(1, 10, dtype=torch.float32),
        # Missing 'meta_info' key
    }
    
    debug_outputs = {'test': {'data': torch.randn(2, 2)}}
    
    # This should raise an assertion error but not crash the buffer worker
    with pytest.raises(AssertionError):
        debugger.add_to_buffer(debug_outputs, malformed_datapoint)


def test_buffer_reset_functionality(sequential_debugger_basic, sample_datapoint):
    """Test buffer reset functionality."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    # Add some data
    debug_outputs = {'test': {'data': torch.randn(3, 3)}}
    debugger.add_to_buffer(debug_outputs, sample_datapoint)
    debugger._buffer_queue.join()
    
    # Verify data was added
    with debugger._buffer_lock:
        assert len(debugger.current_page_data) == 1
        assert debugger.current_page_size > 0
    
    # Reset buffer
    debugger.reset_buffer()
    
    # Verify buffer was reset
    with debugger._buffer_lock:
        assert len(debugger.current_page_data) == 0
        assert debugger.current_page_size == 0
        assert debugger.current_page_idx == 0
