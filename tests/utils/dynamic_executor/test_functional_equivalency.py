from typing import List, Dict, Any
import pytest
import time
import numpy as np
from utils.dynamic_executor import create_dynamic_executor


def simple_function(x: int) -> int:
    """Simple function for testing - just returns x * 2."""
    return x * 2


def computation_function(x: int) -> Dict[str, Any]:
    """More complex function that simulates evaluation workload."""
    # Simulate some computation
    result = 0
    for i in range(x * 1000):
        result += np.sin(i) * np.cos(i)

    return {'input': x, 'result': result, 'squared': x * x, 'timestamp': time.time()}


def error_function(x: int) -> int:
    """Function that raises an error at specific input values."""
    if x == 7:
        raise ValueError(f"Test error at input {x}")
    return x * 3


def intermittent_error_function(x: int) -> int:
    """Function that raises different types of errors at different inputs."""
    if x == 3:
        raise RuntimeError(f"Runtime error at input {x}")
    elif x == 8:
        raise KeyError(f"Key error at input {x}")
    return x + 10


def test_basic_equivalency():
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


def test_complex_computation_equivalency():
    """Test equivalency with more complex computation function."""
    inputs = list(range(5))

    # Sequential execution
    sequential_results = []
    for x in inputs:
        sequential_results.append(computation_function(x))

    # Dynamic executor execution
    executor = create_dynamic_executor(max_workers=3, min_workers=1)
    with executor:
        dynamic_results = list(executor.map(computation_function, inputs))

    # Check that inputs and core computations match
    assert len(dynamic_results) == len(sequential_results)
    for i, (seq_result, dynamic_result) in enumerate(
        zip(sequential_results, dynamic_results)
    ):
        assert seq_result['input'] == dynamic_result['input'] == i
        assert seq_result['squared'] == dynamic_result['squared'] == i * i
        # Results should be very close (floating point computation)
        assert abs(seq_result['result'] - dynamic_result['result']) < 1e-10


def test_result_ordering():
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


def test_deterministic_computation():
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


def test_large_dataset_equivalency():
    """Test equivalency with larger dataset to stress test the system."""
    inputs = list(range(50))

    # Sequential execution
    sequential_results = [simple_function(x) for x in inputs]

    # Dynamic executor execution
    executor = create_dynamic_executor(max_workers=6)
    with executor:
        dynamic_results = list(executor.map(simple_function, inputs))

    assert len(dynamic_results) == len(sequential_results) == 50
    assert dynamic_results == sequential_results


def test_map_order_preservation_with_timeout():
    """Test that map preserves order even with timeout scenarios."""

    def variable_time_function(x: int) -> int:
        # Make later items take less time to test ordering
        sleep_time = 0.1 / (x + 1)
        time.sleep(sleep_time)
        return x * 3

    inputs = list(range(10))
    expected = [x * 3 for x in inputs]

    executor = create_dynamic_executor(max_workers=4)
    with executor:
        results = list(executor.map(variable_time_function, inputs, timeout=5.0))

    assert results == expected


def test_map_empty_iterables():
    """Test map method with various empty input scenarios."""
    executor = create_dynamic_executor(max_workers=2)

    with executor:
        # Empty iterables
        results = list(executor.map(simple_function, []))
        assert results == []

        # Multiple empty iterables
        results = list(executor.map(lambda x, y: x + y, [], []))
        assert results == []


def test_error_equivalency_single_error():
    """Test that dynamic executor and for loop behave identically when an error occurs."""
    inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # error_function will fail at x=7

    # Sequential execution (for loop) - should raise ValueError at x=7
    sequential_error = None
    sequential_results = []
    try:
        for x in inputs:
            sequential_results.append(error_function(x))
    except Exception as e:
        sequential_error = e
        sequential_error_type = type(e)
        sequential_error_message = str(e)

    # Dynamic executor execution - should also raise ValueError at x=7 and stop immediately
    executor_error = None
    with pytest.raises(Exception) as exec_info:
        executor = create_dynamic_executor(max_workers=3)
        with executor:
            list(executor.map(error_function, inputs))

    executor_error = exec_info.value
    executor_error_type = type(executor_error)
    executor_error_message = str(executor_error)

    # Both should have raised an error
    assert (
        sequential_error is not None
    ), "Sequential execution should have raised an error"
    assert executor_error is not None, "Dynamic executor should have raised an error"

    # Error types and messages should be the same
    assert (
        sequential_error_type == executor_error_type
    ), f"Error types differ: {sequential_error_type} vs {executor_error_type}"
    assert (
        sequential_error_message == executor_error_message
    ), f"Error messages differ: '{sequential_error_message}' vs '{executor_error_message}'"


def test_error_equivalency_multiple_potential_errors():
    """Test that both execution methods fail on the first error encountered."""
    inputs = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]  # intermittent_error_function fails at x=3 and x=8

    # Sequential execution - should fail at x=3 (first error)
    sequential_error = None
    try:
        for x in inputs:
            intermittent_error_function(x)
    except Exception as e:
        sequential_error = e

    # Dynamic executor execution - should also fail at x=3 (first error)
    executor_error = None
    with pytest.raises(Exception) as exec_info:
        executor = create_dynamic_executor(max_workers=2)
        with executor:
            list(executor.map(intermittent_error_function, inputs))

    executor_error = exec_info.value

    # Both should have raised the same error (RuntimeError at x=3)
    assert type(sequential_error) == type(executor_error) == RuntimeError
    assert "Runtime error at input 3" in str(sequential_error)
    assert "Runtime error at input 3" in str(executor_error)


def test_error_equivalency_no_partial_results():
    """Test that both methods provide no partial results when an error occurs."""
    inputs = [10, 20, 30, 7, 40, 50]  # error at index 3 (x=7)

    # Sequential execution - collect results until error
    sequential_partial_results = []
    sequential_error = None
    try:
        for x in inputs:
            result = error_function(x)
            sequential_partial_results.append(result)
    except Exception as e:
        sequential_error = e

    # Dynamic executor execution - should fail completely
    executor_error = None
    executor_partial_results = []
    try:
        executor = create_dynamic_executor(max_workers=2)
        with executor:
            executor_partial_results = list(executor.map(error_function, inputs))
    except Exception as e:
        executor_error = e

    # Both should have errors
    assert sequential_error is not None
    assert executor_error is not None

    # Sequential would have partial results (before error), but executor should have none
    # This demonstrates fail-fast behavior - no partial results returned
    assert len(sequential_partial_results) == 3  # [30, 60, 90] before error at index 3
    assert len(executor_partial_results) == 0  # Fail-fast: no partial results

    # Error types should be the same
    assert type(sequential_error) == type(executor_error) == ValueError


def test_error_equivalency_early_vs_late_error():
    """Test error behavior when error occurs early vs late in the dataset."""

    # Test early error (index 1)
    inputs_early_error = [1, 7, 3, 4, 5]  # error at index 1

    # Sequential execution
    with pytest.raises(ValueError, match="Test error at input 7"):
        for x in inputs_early_error:
            error_function(x)

    # Dynamic executor execution
    with pytest.raises(ValueError, match="Test error at input 7"):
        executor = create_dynamic_executor(max_workers=3)
        with executor:
            list(executor.map(error_function, inputs_early_error))

    # Test late error (index 4)
    inputs_late_error = [1, 2, 3, 4, 7]  # error at index 4

    # Sequential execution
    with pytest.raises(ValueError, match="Test error at input 7"):
        for x in inputs_late_error:
            error_function(x)

    # Dynamic executor execution
    with pytest.raises(ValueError, match="Test error at input 7"):
        executor = create_dynamic_executor(max_workers=3)
        with executor:
            list(executor.map(error_function, inputs_late_error))
