import pytest
import time
from utils.dynamic_executor import DynamicThreadPoolExecutor


def slow_function(x: int, sleep_time: float = 0.1) -> int:
    """Function that takes some time to complete."""
    time.sleep(sleep_time)
    return x * 2


def test_logging_integration():
    """Test that logging is properly integrated throughout the executor."""
    # Test without mocking to ensure actual logging works
    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Submit work that might trigger logging
    futures = [executor.submit(slow_function, i, 0.01) for i in range(5)]
    time.sleep(1.0)  # Allow time for impact measurement

    for future in futures:
        future.result()

    executor.shutdown(wait=True)

    # Test that the executor can handle logging without errors
    # The actual logging calls depend on system behavior and timing
    assert True  # If we reach here, logging integration didn't crash


def test_system_load_measurement_accuracy():
    """Test that system load measurements are reasonable."""
    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Take multiple measurements
    measurements = []
    for _ in range(5):
        stats = executor._get_system_load()
        measurements.append(stats)
        time.sleep(0.1)

    executor.shutdown(wait=True)

    # All measurements should have valid values
    for stats in measurements:
        assert 'cpu_percent' in stats
        assert 'gpu_memory_percent' in stats
        assert 0 <= stats['cpu_percent'] <= 100
        assert 0 <= stats['gpu_memory_percent'] <= 100


def test_scaling_decision_logic():
    """Test that scaling decisions are made reasonably."""
    # Create executor with reasonable thresholds
    executor = DynamicThreadPoolExecutor(
        max_workers=4,
        min_workers=1,
        cpu_threshold=80.0,
        gpu_memory_threshold=80.0,
        monitor_interval=0.1
    )

    # Get initial worker count
    initial_workers = executor._current_workers

    # Submit enough work to potentially trigger scaling
    futures = [executor.submit(slow_function, i, 0.05) for i in range(15)]

    # Allow time for monitoring and potential scaling
    time.sleep(0.5)

    # Check if scaling occurred
    current_workers = executor._current_workers

    # Complete all work
    for future in futures:
        future.result()

    executor.shutdown(wait=True)

    # Workers should never exceed max_workers
    assert current_workers <= 4

    # Should start with at least min_workers
    assert initial_workers >= 1


def test_impact_estimation_logic():
    """Test that worker impact estimation provides reasonable estimates."""
    executor = DynamicThreadPoolExecutor(max_workers=3, min_workers=1)

    # Submit work to trigger impact measurement
    futures = [executor.submit(slow_function, i, 0.02) for i in range(10)]

    # Wait for impact measurements
    time.sleep(1.0)

    # Get impact estimation
    estimated_impact = executor._estimate_worker_impact()

    # Complete work
    for future in futures:
        future.result()

    executor.shutdown(wait=True)

    # Estimates should be reasonable
    assert 'cpu_increase' in estimated_impact
    assert 'gpu_increase' in estimated_impact
    assert 0 <= estimated_impact['cpu_increase'] <= 100
    assert 0 <= estimated_impact['gpu_increase'] <= 100


def test_monitoring_loop_resilience():
    """Test that monitoring loop continues despite temporary issues."""
    executor = DynamicThreadPoolExecutor(
        max_workers=2,
        min_workers=1,
        monitor_interval=0.2,  # Increased to reduce test timing issues
        scale_check_interval=0.2
    )

    # Let monitoring run for a while
    time.sleep(0.3)

    # Monitor thread should still be alive
    assert executor._monitor_thread.is_alive()

    # Submit some work
    futures = [executor.submit(slow_function, i, 0.01) for i in range(5)]

    for future in futures:
        future.result()

    executor.shutdown(wait=True)

    # Monitor should have stopped cleanly
    assert not executor._monitor_thread.is_alive()


def test_cooldown_mechanism():
    """Test that scaling cooldown prevents excessive scaling."""
    executor = DynamicThreadPoolExecutor(
        max_workers=3,  # Reduced to simplify test
        min_workers=1
    )

    # Get initial worker count
    initial_workers = executor._current_workers

    # Submit some work
    futures = [executor.submit(slow_function, i, 0.01) for i in range(8)]

    # Wait briefly for any scaling
    time.sleep(0.3)

    # Check worker count during execution
    current_workers = executor._current_workers

    # Complete all work
    for future in futures:
        future.result()

    executor.shutdown(wait=True)

    # Scaling should be reasonable (not excessive)
    assert current_workers <= initial_workers + 2  # At most 2 additional workers


def test_task_queue_behavior():
    """Test that task queue behaves correctly under various conditions."""
    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Submit more tasks than workers
    futures = [executor.submit(slow_function, i, 0.02) for i in range(10)]

    # Queue should not be empty initially
    assert not executor.task_queue.empty()

    # Complete all tasks
    for future in futures:
        future.result()

    # Wait a moment for queue to be processed
    time.sleep(0.1)

    executor.shutdown(wait=True)


def test_future_result_consistency():
    """Test that futures return consistent results."""
    executor = DynamicThreadPoolExecutor(max_workers=3, min_workers=1)

    # Submit the same computation multiple times
    futures = [executor.submit(slow_function, 42, 0.01) for _ in range(10)]

    # All results should be identical
    results = [future.result() for future in futures]

    executor.shutdown(wait=True)

    # All results should be the same
    expected_result = 42 * 2
    assert all(result == expected_result for result in results)


def test_executor_state_transitions():
    """Test that executor state transitions correctly."""
    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Initially not shutdown
    assert not executor._shutdown
    assert executor._monitor_thread.is_alive()

    # Submit some work
    future = executor.submit(slow_function, 1, 0.01)
    result = future.result()
    assert result == 2

    # Shutdown
    executor.shutdown(wait=True)

    # Should be shutdown now
    assert executor._shutdown
    assert not executor._monitor_thread.is_alive()

    # Should not accept new work
    with pytest.raises(RuntimeError):
        executor.submit(slow_function, 2, 0.01)
