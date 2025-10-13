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


def slow_function(x: int, sleep_time: float = 0.01) -> int:
    """Function that takes some time to complete."""
    import time

    time.sleep(sleep_time)
    return x * 2


def test_error_propagation():
    """Test that dynamic executor fails fast on errors, just like a for loop."""
    inputs = [1, 2, 3, 4, 5, 6]  # input 5 will cause an error

    # Sequential execution - should raise ValueError
    with pytest.raises(ValueError, match="Test error for input 5"):
        for x in inputs:
            error_function(x)

    # Executor execution - should also raise ValueError and stop immediately
    executor = create_dynamic_executor(max_workers=3)
    with executor:
        with pytest.raises(ValueError, match="Test error for input 5"):
            list(executor.map(error_function, inputs))


def test_error_logging_integration():
    """Test that errors are properly logged instead of printed."""

    def error_function():
        raise ValueError("Test error")

    with patch('logging.error') as mock_error:
        executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

        future = executor.submit(error_function)
        with pytest.raises(ValueError):
            future.result()

        executor.shutdown(wait=True)

        # Check that error was logged (fail-fast logs with ERROR level)
        mock_error.assert_called()


def test_fail_fast_behavior():
    """Test that executor stops immediately on first error (fail-fast)."""

    def intermittent_error_function(x: int) -> int:
        if x == 3:
            raise RuntimeError("Intermittent error")
        return x * 2

    executor = DynamicThreadPoolExecutor(max_workers=3, min_workers=1)

    # Submit tasks that include an error
    with pytest.raises(RuntimeError, match="Intermittent error"):
        list(executor.map(intermittent_error_function, range(6)))

    # After error, executor should refuse new submissions (failed due to error)
    with pytest.raises(RuntimeError, match="failed due to a previous task error"):
        executor.submit(intermittent_error_function, 1)

    executor.shutdown(wait=True)


def test_fail_fast_during_execution():
    """Test that executor fails fast even during concurrent execution."""

    def slow_error_function(x: int) -> int:
        time.sleep(0.02)  # Small delay to allow concurrent execution
        if x == 2:
            raise ValueError(f"Error for {x}")
        return x * 2

    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Submit tasks that will fail on the third one
    with pytest.raises(ValueError, match="Error for 2"):
        list(executor.map(slow_error_function, range(10)))

    # Should have failed and shutdown
    assert executor._failed is True

    # Let execution complete
    executor.shutdown(wait=True)
    assert executor._shutdown is True


def test_gpu_error_handling():
    """Test that GPU monitoring errors are handled gracefully in fail-fast mode."""
    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Test that _get_system_load handles GPU errors gracefully
    with patch(
        'torch.cuda.memory_allocated', side_effect=RuntimeError("Mock GPU error")
    ):
        # This should not raise an exception - GPU errors in monitoring should be handled
        stats = executor._get_system_load()
        assert 'cpu_percent' in stats
        assert 'gpu_memory_percent' in stats
        assert stats['gpu_memory_percent'] == 0.0  # Should default to 0 on error

    # Executor should still be functional after GPU monitoring error
    future = executor.submit(slow_function, 1, 0.01)
    assert future.result() == 2

    executor.shutdown(wait=True)


def test_invalid_function_submission():
    """Test fail-fast behavior when submitting invalid functions."""
    executor = DynamicThreadPoolExecutor(max_workers=2, min_workers=1)

    # Submit a function that doesn't exist - should trigger fail-fast
    future = executor.submit(None)  # This will cause TypeError in worker

    with pytest.raises(TypeError):
        future.result()

    # After error, executor should be failed and refuse new submissions
    assert executor._failed is True
    with pytest.raises(RuntimeError, match="failed due to a previous task error"):
        executor.submit(slow_function, 1, 0.01)

    executor.shutdown(wait=True)


def test_timeout_error_scenarios():
    """Test timeout handling in fail-fast mode."""

    def very_slow_function(x: int) -> int:
        time.sleep(0.2)
        return x * 2

    executor = create_dynamic_executor(max_workers=2)

    # Test with very short timeout - should handle timeout gracefully
    with executor:
        # This should timeout due to very short timeout (0.01s) vs slow function (0.2s)
        with pytest.raises(
            (TimeoutError, Exception)
        ):  # Handle timeout or concurrent.futures timeout
            list(executor.map(very_slow_function, range(5), timeout=0.01))

    # Executor should still shutdown cleanly after timeout
    assert executor._shutdown is True
