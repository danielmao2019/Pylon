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

    return {
        'input': x,
        'result': result,
        'squared': x * x,
        'timestamp': time.time()
    }


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
    for i, (seq_result, dynamic_result) in enumerate(zip(sequential_results, dynamic_results)):
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
