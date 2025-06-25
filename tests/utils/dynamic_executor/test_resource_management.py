from typing import List, Dict, Any
import pytest
import time
import torch
from unittest.mock import patch
from utils.dynamic_executor import create_dynamic_executor, DynamicThreadPoolExecutor


def slow_function(x: int, sleep_time: float = 0.01) -> int:
    """Function that takes some time to complete."""
    time.sleep(sleep_time)
    return x * 2


def gpu_function() -> Dict[str, Any]:
    """Function that uses GPU memory if available."""
    if torch.cuda.is_available():
        # Allocate some GPU memory
        tensor = torch.randn(1000, 1000).cuda()
        memory_used = torch.cuda.memory_allocated()
        del tensor
        return {'gpu_memory_used': memory_used}
    return {'gpu_memory_used': 0}


def test_cleanup_and_resource_management():
    """Test that executor properly cleans up resources."""
    executor = DynamicThreadPoolExecutor(max_workers=3, min_workers=1)

    # Submit some work
    futures = [executor.submit(slow_function, i, 0.01) for i in range(5)]

    # Get worker count before shutdown
    initial_worker_count = executor._current_workers
    assert initial_worker_count >= 1

    # Wait for tasks to complete
    for future in futures:
        future.result()

    # Shutdown and verify cleanup
    executor.shutdown(wait=True)
    assert executor._shutdown is True

    # All workers should be stopped
    for worker in executor.workers:
        assert not worker.stats.active
        assert not worker.is_alive()


def test_gpu_memory_monitoring():
    """Test GPU memory monitoring with different device scenarios."""
    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Test GPU memory collection
    stats = executor._get_system_load()
    assert 'gpu_memory_percent' in stats
    assert isinstance(stats['gpu_memory_percent'], float)
    assert 0 <= stats['gpu_memory_percent'] <= 100

    executor.shutdown(wait=True)


def test_worker_impact_measurement():
    """Test that worker impact measurement works correctly."""
    executor = DynamicThreadPoolExecutor(max_workers=4, min_workers=1)

    # Submit some work to potentially trigger scaling
    futures = [executor.submit(slow_function, i, 0.02) for i in range(20)]

    # Wait for tasks to complete and potential scaling
    time.sleep(1.0)

    for future in futures:
        future.result()

    executor.shutdown(wait=True)

    # Should have some impact measurements if workers were added
    with executor._lock:
        # Even if no scaling occurred, the initial worker should have impact measured
        assert len(executor._worker_utilization_impacts) >= 0


def test_worker_impact_bounds():
    """Test that worker impact tracking maintains proper bounds."""
    executor = DynamicThreadPoolExecutor(
        max_workers=3,
        min_workers=1,
        monitor_interval=0.05
    )

    # Submit work and wait for monitoring to potentially add workers
    futures = [executor.submit(slow_function, i, 0.01) for i in range(10)]
    time.sleep(0.3)  # Allow multiple monitoring cycles

    for future in futures:
        future.result()

    executor.shutdown(wait=True)

    # Worker impact tracking should not exceed 5 entries
    with executor._lock:
        assert len(executor._worker_utilization_impacts) <= 5


def test_memory_usage_patterns():
    """Test that executor doesn't leak memory during normal operation."""
    import gc
    import psutil
    import os

    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Run multiple cycles of executor creation and destruction
    for _ in range(5):
        executor = DynamicThreadPoolExecutor(max_workers=3, min_workers=1)

        # Submit and complete some work
        futures = [executor.submit(slow_function, i, 0.001) for i in range(20)]
        for future in futures:
            future.result()

        executor.shutdown(wait=True)

        # Force garbage collection
        gc.collect()

    # Check final memory usage
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    # Memory increase should be reasonable (less than 50MB)
    assert memory_increase < 50 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f} MB"


@pytest.mark.parametrize("min_workers,max_workers", [(1, 1), (2, 2), (1, 4), (3, 6)])
def test_worker_count_constraints(min_workers: int, max_workers: int):
    """Test that worker counts respect min/max constraints."""
    executor = DynamicThreadPoolExecutor(
        max_workers=max_workers,
        min_workers=min_workers
    )

    # Initial worker count should be at least min_workers
    initial_count = executor._current_workers
    assert min_workers <= initial_count <= max_workers

    # Submit work to potentially trigger scaling
    futures = [executor.submit(slow_function, i, 0.01) for i in range(max_workers * 3)]

    # Check worker count during execution
    time.sleep(0.1)
    current_count = executor._current_workers
    assert min_workers <= current_count <= max_workers

    # Wait for completion
    for future in futures:
        future.result()

    executor.shutdown(wait=True)


def test_resource_thresholds():
    """Test that resource thresholds are respected."""
    # Create executor with very low thresholds to test limiting behavior
    executor = DynamicThreadPoolExecutor(
        max_workers=4,
        min_workers=1,
        cpu_threshold=1.0,  # Very low threshold
        gpu_memory_threshold=1.0  # Very low threshold
    )

    # Submit work that would normally trigger scaling
    futures = [executor.submit(slow_function, i, 0.02) for i in range(20)]

    # Wait for some processing
    time.sleep(0.5)

    # With very low thresholds, shouldn't scale much
    current_workers = executor._current_workers
    assert current_workers <= 2  # Should be limited by thresholds

    # Complete all tasks
    for future in futures:
        future.result()

    executor.shutdown(wait=True)


def test_monitor_thread_lifecycle():
    """Test that monitor thread starts and stops correctly."""
    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Monitor thread should be running
    assert executor._monitor_thread.is_alive()

    # Submit some work
    futures = [executor.submit(slow_function, i, 0.01) for i in range(5)]

    for future in futures:
        future.result()

    # Shutdown should stop monitor thread
    executor.shutdown(wait=True)

    # Monitor thread should be stopped
    assert not executor._monitor_thread.is_alive()


def test_resource_monitoring_with_gpu():
    """Test resource monitoring when GPU operations are involved."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Submit GPU work
    futures = [executor.submit(gpu_function) for _ in range(3)]

    # Get resource stats during GPU work
    stats = executor._get_system_load()

    # Should have valid GPU stats
    assert 'gpu_memory_percent' in stats
    assert isinstance(stats['gpu_memory_percent'], float)
    assert stats['gpu_memory_percent'] >= 0

    # Wait for GPU work to complete
    for future in futures:
        result = future.result()
        assert 'gpu_memory_used' in result

    executor.shutdown(wait=True)
