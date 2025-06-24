from typing import List, Dict, Any
import pytest
import time
import numpy as np
import torch
from utils.dynamic_executor import create_dynamic_executor, DynamicThreadPoolExecutor


def simple_function(x: int) -> int:
    """Simple function for testing - just returns x * 2."""
    return x * 2


def computation_function(x: int) -> Dict[str, Any]:
    """More complex function that simulates evaluation workload."""
    # Simulate some computation
    result = 0
    for i in range(x * 1000):
        result += np.sin(i) * np.cos(i)

    return {
        'input': x,
        'result': result,
        'squared': x * x,
        'timestamp': time.time()
    }


def error_function(x: int) -> int:
    """Function that raises an error for testing error handling."""
    if x == 5:
        raise ValueError(f"Test error for input {x}")
    return x * 3


def test_dynamic_executor_basic_equivalency():
    """Test that dynamic executor produces same results as sequential execution."""
    inputs = list(range(10))

    # Sequential execution (reference)
    sequential_results = []
    for x in inputs:
        sequential_results.append(simple_function(x))

    # Dynamic executor execution
    executor = create_dynamic_executor(max_workers=4, min_workers=1)
    with executor:
        dynamic_results = list(executor.map(simple_function, inputs))

    # Results should be identical (order-preserving)
    assert dynamic_results == sequential_results
    assert len(dynamic_results) == len(sequential_results)
    assert all(a == b for a, b in zip(dynamic_results, sequential_results))


def test_dynamic_executor_complex_equivalency():
    """Test equivalency with more complex computation function."""
    inputs = list(range(5))

    # Sequential execution
    sequential_results = []
    for x in inputs:
        sequential_results.append(computation_function(x))

    # Adaptive executor execution
    executor = create_dynamic_executor(max_workers=3, min_workers=1)
    with executor:
        dynamic_results = list(executor.map(computation_function, inputs))

    # Check that inputs and core computations match
    assert len(dynamic_results) == len(sequential_results)
    for i, (seq_result, adapt_result) in enumerate(zip(sequential_results, dynamic_results)):
        assert seq_result['input'] == adapt_result['input'] == i
        assert seq_result['squared'] == adapt_result['squared'] == i * i
        # Results should be very close (floating point computation)
        assert abs(seq_result['result'] - adapt_result['result']) < 1e-10


def test_dynamic_executor_empty_input():
    """Test behavior with empty input list."""
    inputs = []

    # Sequential execution
    sequential_results = [simple_function(x) for x in inputs]

    # Adaptive executor execution
    executor = create_dynamic_executor(max_workers=2)
    with executor:
        dynamic_results = list(executor.map(simple_function, inputs))

    assert dynamic_results == sequential_results == []


def test_dynamic_executor_single_input():
    """Test behavior with single input."""
    inputs = [42]

    # Sequential execution
    sequential_results = [simple_function(x) for x in inputs]

    # Adaptive executor execution
    executor = create_dynamic_executor(max_workers=4)
    with executor:
        dynamic_results = list(executor.map(simple_function, inputs))

    assert dynamic_results == sequential_results == [84]


def test_dynamic_executor_error_handling():
    """Test that dynamic executor handles errors the same way as sequential execution."""
    inputs = [1, 2, 3, 4, 5, 6]  # input 5 will cause an error

    # Sequential execution - should raise ValueError
    with pytest.raises(ValueError, match="Test error for input 5"):
        for x in inputs:
            error_function(x)

    # Adaptive executor execution - should also raise ValueError
    executor = create_dynamic_executor(max_workers=3)
    with executor:
        with pytest.raises(ValueError):
            list(executor.map(error_function, inputs))


def test_dynamic_executor_result_ordering():
    """Test that results maintain correct ordering even with parallel execution."""
    inputs = list(range(20))
    expected_results = [x * 2 for x in inputs]

    # Run multiple times to catch any ordering issues
    for _ in range(3):
        executor = create_dynamic_executor(max_workers=5)
        with executor:
            dynamic_results = list(executor.map(simple_function, inputs))

        assert dynamic_results == expected_results
        # Verify each result is in the correct position
        for i, result in enumerate(dynamic_results):
            assert result == inputs[i] * 2


def test_dynamic_executor_deterministic_computation():
    """Test that deterministic computations produce identical results."""
    inputs = [10, 20, 30]

    # Run the same computation multiple times
    all_results = []
    for _ in range(3):
        executor = create_dynamic_executor(max_workers=2)
        with executor:
            results = list(executor.map(computation_function, inputs))
        all_results.append(results)

    # All runs should produce identical results for deterministic computation
    for i in range(len(inputs)):
        first_run_result = all_results[0][i]['result']
        for run_idx in range(1, len(all_results)):
            assert abs(all_results[run_idx][i]['result'] - first_run_result) < 1e-10


def test_dynamic_executor_submit_method():
    """Test the submit method works equivalently to map."""
    inputs = [1, 2, 3, 4, 5]

    # Sequential execution
    sequential_results = [simple_function(x) for x in inputs]

    # Adaptive executor with submit
    executor = create_dynamic_executor(max_workers=3)
    with executor:
        futures = [executor.submit(simple_function, x) for x in inputs]
        dynamic_results = [f.result() for f in futures]

    assert dynamic_results == sequential_results


def test_dynamic_thread_pool_executor_direct():
    """Test DynamicThreadPoolExecutor class directly."""
    inputs = [1, 2, 3, 4]
    expected = [2, 4, 6, 8]

    # Test with direct class usage
    with DynamicThreadPoolExecutor(max_workers=2, min_workers=1) as executor:
        results = list(executor.map(simple_function, inputs))

    assert results == expected


def test_dynamic_executor_worker_count_behavior():
    """Test that dynamic executor respects worker count constraints."""
    executor_instance = DynamicThreadPoolExecutor(max_workers=4, min_workers=1)

    # Worker count should be within bounds
    with executor_instance._lock:
        current_count = len(executor_instance.workers)
    assert 1 <= current_count <= 4

    # Initial count should be reasonable (not more than max_workers)
    assert current_count <= 4


@pytest.mark.parametrize("max_workers", [1, 2, 4, 8])
def test_dynamic_executor_different_worker_counts(max_workers: int):
    """Test that different worker counts all produce equivalent results."""
    inputs = list(range(10))
    expected = [x * 2 for x in inputs]

    executor = create_dynamic_executor(max_workers=max_workers)
    with executor:
        results = list(executor.map(simple_function, inputs))

    assert results == expected


def test_dynamic_executor_large_dataset():
    """Test equivalency with larger dataset to stress test the system."""
    inputs = list(range(50))

    # Sequential execution
    sequential_results = [simple_function(x) for x in inputs]

    # Adaptive executor execution
    executor = create_dynamic_executor(max_workers=6)
    with executor:
        dynamic_results = list(executor.map(simple_function, inputs))

    assert len(dynamic_results) == len(sequential_results) == 50
    assert dynamic_results == sequential_results


def test_dynamic_executor_context_manager():
    """Test that context manager properly handles cleanup."""
    inputs = [1, 2, 3]

    with create_dynamic_executor(max_workers=2) as executor:
        results = list(executor.map(simple_function, inputs))

    assert results == [2, 4, 6]
    # After context exit, executor should be shutdown
    # Note: We can't easily test this without accessing private attributes
