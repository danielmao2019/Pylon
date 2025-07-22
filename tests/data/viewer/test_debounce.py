"""Unit tests for the debounce decorator."""

import time
import threading
from unittest.mock import Mock

import pytest
from dash.exceptions import PreventUpdate

from data.viewer.utils.debounce import debounce


def test_single_call_executes_immediately():
    """Test that a single call executes immediately when no previous calls."""
    mock = Mock()
    
    @debounce
    def test_func(value):
        mock(value)
        return f"result_{value}"
    
    # Call function once - should execute immediately
    result = test_func(42)
    
    # Should execute immediately with correct result
    assert mock.call_count == 1
    mock.assert_called_with(42)
    assert result == "result_42"


def test_multiple_rapid_calls_with_last_call_wins():
    """Test that rapid calls schedule the last call to execute after delay."""
    mock = Mock()
    
    @debounce
    def test_func(value):
        mock(value)
        return f"result_{value}"
    
    # First call should execute immediately
    result1 = test_func(1)
    assert result1 == "result_1"
    assert mock.call_count == 1
    
    # Rapid subsequent calls should raise PreventUpdate but schedule last one
    for i in range(2, 6):
        with pytest.raises(PreventUpdate):
            test_func(i)
        time.sleep(0.05)  # 50ms between calls (less than 1s debounce)
    
    # Should still only have 1 execution (first call)
    assert mock.call_count == 1
    mock.assert_called_with(1)
    
    # Wait for delayed execution of last call
    time.sleep(1.2)
    
    # Last call (5) should have executed
    assert mock.call_count == 2
    mock.assert_any_call(1)  # First call
    mock.assert_any_call(5)  # Last call executed after delay


def test_calls_allowed_after_delay():
    """Test that calls are allowed after the debounce delay period."""
    mock = Mock()
    
    @debounce
    def test_func(value):
        mock(value)
        return f"result_{value}"
    
    # First call - should execute
    result1 = test_func(1)
    assert result1 == "result_1"
    assert mock.call_count == 1
    
    # Wait longer than debounce period
    time.sleep(1.1)
    
    # Second call after delay - should execute
    result2 = test_func(2)
    assert result2 == "result_2"
    assert mock.call_count == 2
    
    # Verify both calls happened with correct arguments
    mock.assert_any_call(1)
    mock.assert_any_call(2)


def test_argument_preservation():
    """Test that function arguments and keyword arguments are preserved."""
    mock = Mock()
    
    @debounce
    def test_func(a, b, c=None, d=None):
        mock(a, b, c, d)
        return f"{a}-{b}-{c}-{d}"
    
    # Call with various arguments
    result = test_func(1, 2, c=3, d=4)
    
    # Should be called with correct arguments and return correct result
    assert mock.call_count == 1
    mock.assert_called_with(1, 2, 3, 4)
    assert result == "1-2-3-4"


def test_concurrent_functions():
    """Test that multiple debounced functions work independently."""
    mock1 = Mock()
    mock2 = Mock()
    
    @debounce
    def func1(value):
        mock1(value)
        return f"func1_{value}"
    
    @debounce
    def func2(value):
        mock2(value)
        return f"func2_{value}"
    
    # Call both functions - both should execute
    result1 = func1("a")
    result2 = func2("b")
    
    # Both should execute independently
    assert mock1.call_count == 1
    assert mock2.call_count == 1
    mock1.assert_called_with("a")
    mock2.assert_called_with("b")
    assert result1 == "func1_a"
    assert result2 == "func2_b"


def test_function_metadata_preserved():
    """Test that the decorated function preserves the original function's metadata."""
    @debounce
    def test_func(x, y):
        """Test function docstring."""
        return x + y
    
    # Check metadata is preserved
    assert test_func.__name__ == "test_func"
    assert test_func.__doc__ == "Test function docstring."


def test_rapid_then_delayed_execution():
    """Test behavior with rapid calls followed by delayed call."""
    mock = Mock()
    
    @debounce
    def test_func(value):
        mock(value)
        return f"result_{value}"
    
    # First call - executes immediately
    result1 = test_func(1)
    assert result1 == "result_1"
    assert mock.call_count == 1
    
    # Rapid calls - all prevented but last one scheduled
    for i in range(2, 6):
        with pytest.raises(PreventUpdate):
            test_func(i)
        time.sleep(0.1)
    
    # Still only first execution
    assert mock.call_count == 1
    
    # Wait for delayed execution of last call (5)
    time.sleep(1.2)
    
    # Should have 2 executions now (first + delayed last)
    assert mock.call_count == 2
    mock.assert_any_call(1)
    mock.assert_any_call(5)
    
    # Wait for debounce period to pass
    time.sleep(1.1)
    
    # New call after delay - should execute immediately
    result3 = test_func(10)
    assert result3 == "result_10"
    assert mock.call_count == 3
    mock.assert_any_call(10)


def test_debounce_timing_precision():
    """Test that the 1-second debounce timing is enforced correctly."""
    mock = Mock()
    
    @debounce
    def test_func():
        mock()
        return "executed"
    
    # First call - executes immediately
    test_func()
    assert mock.call_count == 1
    
    # Call at 0.9 seconds - should be prevented and scheduled
    time.sleep(0.9)
    with pytest.raises(PreventUpdate):
        test_func()
    assert mock.call_count == 1
    
    # Wait for scheduled execution
    time.sleep(0.3)  # Allow delayed execution to complete
    assert mock.call_count == 2  # Delayed execution should have happened
    
    # Call at sufficient delay - should execute immediately  
    time.sleep(1.1)
    test_func()
    assert mock.call_count == 3


def test_thread_safety_basic():
    """Test basic thread safety - concurrent functions work independently."""
    mock1 = Mock()
    mock2 = Mock()
    
    @debounce
    def func1():
        mock1()
        return "func1_result"
    
    @debounce  
    def func2():
        mock2()
        return "func2_result"
    
    def thread1_work():
        func1()
    
    def thread2_work():
        func2()
    
    # Create threads for different functions
    thread1 = threading.Thread(target=thread1_work)
    thread2 = threading.Thread(target=thread2_work)
    
    # Start both threads
    thread1.start()
    thread2.start()
    
    # Wait for completion
    thread1.join()
    thread2.join()
    
    # Both functions should execute independently
    assert mock1.call_count == 1
    assert mock2.call_count == 1