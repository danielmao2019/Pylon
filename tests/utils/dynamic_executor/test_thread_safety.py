from typing import List, Dict, Any
import pytest
import time
import threading
from utils.dynamic_executor import create_dynamic_executor, DynamicThreadPoolExecutor


def slow_function(x: int, sleep_time: float = 0.01) -> int:
    """Function that takes some time to complete."""
    time.sleep(sleep_time)
    return x * 2


def test_pending_tasks_thread_safety():
    """Test that pending tasks counter is thread-safe."""
    executor = DynamicThreadPoolExecutor(max_workers=4, min_workers=1)
    
    # Submit tasks and collect all futures
    all_futures = []
    
    def submit_batch(batch_id: int, num_tasks: int):
        batch_futures = []
        for i in range(num_tasks):
            future = executor.submit(slow_function, batch_id * 100 + i, 0.01)
            batch_futures.append(future)
        all_futures.extend(batch_futures)
    
    # Submit from multiple threads
    threads = []
    for batch_id in range(3):
        thread = threading.Thread(target=submit_batch, args=[batch_id, 10])
        threads.append(thread)
        thread.start()
    
    # Wait for all submissions to complete
    for thread in threads:
        thread.join()
    
    # Wait for all tasks to complete by collecting results
    for future in all_futures:
        future.result()
    
    # Now shutdown and verify counter
    executor.shutdown(wait=True)
    
    # After all tasks complete and shutdown, pending should be 0
    assert executor._pending_tasks == 0


def test_concurrent_submissions():
    """Test concurrent task submissions from multiple threads."""
    executor = DynamicThreadPoolExecutor(max_workers=4, min_workers=1)
    
    results = []
    results_lock = threading.Lock()
    
    def submit_and_collect(start_idx: int, count: int):
        local_results = []
        for i in range(start_idx, start_idx + count):
            future = executor.submit(slow_function, i, 0.01)
            local_results.append((i, future.result()))
        
        with results_lock:
            results.extend(local_results)
    
    # Submit from multiple threads
    threads = []
    for thread_idx in range(3):
        thread = threading.Thread(
            target=submit_and_collect, 
            args=[thread_idx * 10, 10]
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    executor.shutdown(wait=True)
    
    # Verify all results are correct
    assert len(results) == 30
    for input_val, output_val in results:
        assert output_val == input_val * 2


def test_concurrent_map_operations():
    """Test multiple map operations running concurrently."""
    executor = DynamicThreadPoolExecutor(max_workers=4, min_workers=1)
    
    results = []
    results_lock = threading.Lock()
    
    def run_map_operation(thread_id: int, data_range: range):
        local_results = list(executor.map(slow_function, data_range))
        with results_lock:
            results.append((thread_id, local_results))
    
    # Run multiple map operations concurrently
    threads = []
    for thread_id in range(3):
        data_range = range(thread_id * 5, (thread_id + 1) * 5)
        thread = threading.Thread(target=run_map_operation, args=[thread_id, data_range])
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    executor.shutdown(wait=True)
    
    # Verify results
    assert len(results) == 3
    for thread_id, thread_results in results:
        expected_start = thread_id * 5
        expected_results = [i * 2 for i in range(expected_start, expected_start + 5)]
        assert thread_results == expected_results


def test_submit_during_shutdown():
    """Test behavior when submitting tasks during shutdown."""
    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)
    
    # Submit some initial work
    initial_futures = [executor.submit(slow_function, i, 0.02) for i in range(5)]
    
    # Start shutdown in a separate thread
    def delayed_shutdown():
        time.sleep(0.05)  # Let some tasks start
        executor.shutdown(wait=False)
    
    shutdown_thread = threading.Thread(target=delayed_shutdown)
    shutdown_thread.start()
    
    # Try to submit more work (some might succeed, some might fail)
    late_futures = []
    for i in range(5, 10):
        try:
            future = executor.submit(slow_function, i, 0.01)
            late_futures.append(future)
        except RuntimeError:
            # Expected if executor is shutting down
            pass
    
    # Wait for shutdown to complete
    shutdown_thread.join()
    executor.shutdown(wait=True)  # Ensure full shutdown
    
    # Collect results from initial futures
    for future in initial_futures:
        future.result()  # Should complete successfully
    
    # Late futures might have completed or been cancelled
    for future in late_futures:
        try:
            future.result()
        except RuntimeError:
            # Expected if task was cancelled during shutdown
            pass


def test_worker_state_consistency():
    """Test that worker state remains consistent under concurrent access."""
    executor = DynamicThreadPoolExecutor(max_workers=3, min_workers=1)
    
    # Submit work to potentially trigger worker creation
    futures = [executor.submit(slow_function, i, 0.02) for i in range(15)]
    
    # Check worker count multiple times from different threads
    worker_counts = []
    count_lock = threading.Lock()
    
    def check_worker_count():
        for _ in range(10):
            current_count = executor._current_workers
            with count_lock:
                worker_counts.append(current_count)
            time.sleep(0.001)
    
    # Start multiple threads checking worker count
    check_threads = []
    for _ in range(3):
        thread = threading.Thread(target=check_worker_count)
        check_threads.append(thread)
        thread.start()
    
    # Wait for all tasks and checks to complete
    for future in futures:
        future.result()
    
    for thread in check_threads:
        thread.join()
    
    executor.shutdown(wait=True)
    
    # All worker counts should be within valid bounds
    for count in worker_counts:
        assert 1 <= count <= 3


def test_lock_contention_resilience():
    """Test that the executor handles high lock contention gracefully."""
    executor = DynamicThreadPoolExecutor(max_workers=4, min_workers=1)
    
    # Create high contention by rapidly submitting and checking state
    def rapid_submit_check():
        for i in range(20):
            future = executor.submit(slow_function, i, 0.001)
            current_workers = executor._current_workers  # Accesses lock
            assert current_workers >= 1
            future.result()
    
    # Run multiple threads creating contention
    threads = []
    for _ in range(4):
        thread = threading.Thread(target=rapid_submit_check)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    executor.shutdown(wait=True)
    
    # Should complete without deadlocks or race conditions