from typing import List, Dict, Any
import pytest
import time
import logging
from unittest.mock import patch
from utils.dynamic_executor import create_dynamic_executor, DynamicThreadPoolExecutor


def error_function(x: int) -> int:
    """Function that raises an error for testing error handling."""
    if x == 5:
        raise ValueError(f"Test error for input {x}")
    return x * 3


def test_error_propagation():
    """Test that dynamic executor handles errors the same way as sequential execution."""
    inputs = [1, 2, 3, 4, 5, 6]  # input 5 will cause an error

    # Sequential execution - should raise ValueError
    with pytest.raises(ValueError, match="Test error for input 5"):
        for x in inputs:
            error_function(x)

    # Executor execution - should also raise ValueError
    executor = create_dynamic_executor(max_workers=3)
    with executor:
        with pytest.raises(ValueError):
            list(executor.map(error_function, inputs))


def test_error_logging_integration():
    """Test that errors are properly logged instead of printed."""
    def error_function():
        raise ValueError("Test error")

    with patch('logging.warning') as mock_warning:
        executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

        future = executor.submit(error_function)
        with pytest.raises(ValueError):
            future.result()

        executor.shutdown(wait=True)

        # Check that warning was logged
        mock_warning.assert_called()


def test_worker_error_resilience():
    """Test that worker errors don't crash the entire system."""
    def intermittent_error_function(x: int) -> int:
        if x == 3:
            raise RuntimeError("Intermittent error")
        return x * 2

    executor = DynamicThreadPoolExecutor(max_workers=3, min_workers=1)

    # Submit tasks that include errors
    futures = []
    for i in range(6):
        future = executor.submit(intermittent_error_function, i)
        futures.append((i, future))

    # Collect results and errors
    results = []
    errors = []

    for i, future in futures:
        try:
            result = future.result()
            results.append((i, result))
        except RuntimeError as e:
            errors.append((i, str(e)))

    executor.shutdown(wait=True)

    # Should have 5 successful results and 1 error
    assert len(results) == 5
    assert len(errors) == 1
    assert errors[0][0] == 3  # Error occurred at input 3

    # Verify successful results are correct
    for i, result in results:
        assert result == i * 2


def test_shutdown_with_pending_errors():
    """Test that shutdown works correctly even with pending tasks that error."""
    def slow_error_function(x: int) -> int:
        time.sleep(0.05)  # Small delay
        if x % 2 == 0:
            raise ValueError(f"Error for {x}")
        return x * 2

    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Submit tasks quickly
    futures = [executor.submit(slow_error_function, i) for i in range(10)]

    # Don't wait for results, just shutdown
    executor.shutdown(wait=True)

    # Should have shutdown cleanly
    assert executor._shutdown is True


def test_gpu_error_handling():
    """Test that GPU monitoring errors are handled gracefully."""
    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Test that _get_system_load handles GPU errors gracefully
    with patch('torch.cuda.memory_allocated', side_effect=RuntimeError("Mock GPU error")):
        # This should not raise an exception
        stats = executor._get_system_load()
        assert 'cpu_percent' in stats
        assert 'gpu_memory_percent' in stats
        assert stats['gpu_memory_percent'] == 0.0  # Should default to 0 on error

    executor.shutdown(wait=True)


def test_invalid_function_submission():
    """Test error handling when submitting invalid functions."""
    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Submit a function that doesn't exist
    future = executor.submit(None)  # This should handle the error gracefully

    with pytest.raises(TypeError):
        future.result()

    executor.shutdown(wait=True)


def test_timeout_error_scenarios():
    """Test error handling in timeout scenarios."""
    def slow_function(x: int) -> int:
        time.sleep(0.2)
        return x * 2

    executor = create_dynamic_executor(max_workers=2)

    # Test with very short timeout - should handle timeout gracefully
    with executor:
        try:
            # This might timeout depending on system load
            results = list(executor.map(slow_function, range(5), timeout=0.01))
        except Exception:
            # If timeout occurs, it should be handled gracefully
            pass  # Expected behavior - timeout can occur

    # Executor should still shutdown cleanly
    assert executor._shutdown is True
