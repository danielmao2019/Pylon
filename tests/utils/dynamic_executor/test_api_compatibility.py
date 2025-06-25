from typing import List, Dict, Any
import pytest
import time
import numpy as np
from utils.dynamic_executor import create_dynamic_executor, DynamicThreadPoolExecutor


def simple_function(x: int) -> int:
    """Simple function for testing - just returns x * 2."""
    return x * 2


def test_create_dynamic_executor_factory():
    """Test the factory function creates executor correctly."""
    executor = create_dynamic_executor(max_workers=4, min_workers=2)

    assert executor._max_workers == 4
    assert executor._current_workers >= 2
    assert executor._current_workers <= 4

    executor.shutdown(wait=True)


def test_dynamic_thread_pool_executor_direct():
    """Test DynamicThreadPoolExecutor class directly."""
    inputs = [1, 2, 3, 4]
    expected = [2, 4, 6, 8]

    # Test with direct class usage
    with DynamicThreadPoolExecutor(max_workers=2, min_workers=1) as executor:
        results = list(executor.map(simple_function, inputs))

    assert results == expected


def test_context_manager_interface():
    """Test that context manager properly handles cleanup."""
    inputs = [1, 2, 3]

    with create_dynamic_executor(max_workers=2) as executor:
        results = list(executor.map(simple_function, inputs))

    assert results == [2, 4, 6]
    # After context exit, executor should be shutdown
    assert executor._shutdown is True


def test_submit_method_api():
    """Test the submit method works equivalently to map."""
    inputs = [1, 2, 3, 4, 5]

    # Sequential execution
    sequential_results = [simple_function(x) for x in inputs]

    # Executor with submit
    executor = create_dynamic_executor(max_workers=3)
    with executor:
        futures = [executor.submit(simple_function, x) for x in inputs]
        dynamic_results = [f.result() for f in futures]

    assert dynamic_results == sequential_results


def test_map_method_api():
    """Test map method with various scenarios."""
    inputs = list(range(10))
    expected = [x * 2 for x in inputs]

    executor = create_dynamic_executor(max_workers=4)
    with executor:
        results = list(executor.map(simple_function, inputs))

    assert results == expected


def test_empty_input_handling():
    """Test behavior with empty input list."""
    inputs = []

    # Sequential execution
    sequential_results = [simple_function(x) for x in inputs]

    # Executor execution
    executor = create_dynamic_executor(max_workers=2)
    with executor:
        dynamic_results = list(executor.map(simple_function, inputs))

    assert dynamic_results == sequential_results == []


def test_single_input_handling():
    """Test behavior with single input."""
    inputs = [42]

    # Sequential execution
    sequential_results = [simple_function(x) for x in inputs]

    # Executor execution
    executor = create_dynamic_executor(max_workers=4)
    with executor:
        dynamic_results = list(executor.map(simple_function, inputs))

    assert dynamic_results == sequential_results == [84]


@pytest.mark.parametrize("max_workers", [1, 2, 4, 8])
def test_different_worker_counts(max_workers: int):
    """Test that different worker counts all produce equivalent results."""
    inputs = list(range(10))
    expected = [x * 2 for x in inputs]

    executor = create_dynamic_executor(max_workers=max_workers)
    with executor:
        results = list(executor.map(simple_function, inputs))

    assert results == expected


def test_worker_properties():
    """Test _max_workers and _current_workers properties."""
    executor = DynamicThreadPoolExecutor(max_workers=5, min_workers=2)

    # _max_workers should return the configured maximum
    assert executor._max_workers == 5

    # _current_workers should return actual number of workers
    assert executor._current_workers >= 2  # At least min_workers
    assert executor._current_workers <= 5  # At most max_workers

    executor.shutdown(wait=True)


def test_worker_count_constraints():
    """Test that dynamic executor respects worker count constraints."""
    executor_instance = DynamicThreadPoolExecutor(max_workers=4, min_workers=1)

    # Worker count should be within bounds
    with executor_instance._lock:
        current_count = len(executor_instance.workers)
    assert 1 <= current_count <= 4

    # Initial count should be reasonable (not more than max_workers)
    assert current_count <= 4

    executor_instance.shutdown(wait=True)
