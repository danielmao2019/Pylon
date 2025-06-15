import time
import pytest
from utils.timeout import with_timeout
import threading


def test_timeout_success():
    """Test that a function completing within the timeout period succeeds."""
    @with_timeout(seconds=2)
    def quick_function():
        time.sleep(0.1)
        return "success"

    assert quick_function() == "success"


def test_timeout_failure():
    """Test that a function exceeding the timeout period raises TimeoutError."""
    @with_timeout(seconds=0.1)
    def slow_function():
        time.sleep(0.2)
        return "should not get here"

    with pytest.raises(TimeoutError) as exc_info:
        slow_function()

    assert "timed out after 0.1 seconds" in str(exc_info.value)


def test_timeout_with_exception():
    """Test that exceptions from the wrapped code are properly propagated."""
    @with_timeout(seconds=2)
    def error_function():
        raise ValueError("Test exception")

    with pytest.raises(ValueError) as exc_info:
        error_function()

    assert str(exc_info.value) == "Test exception"


def test_timeout_cleanup():
    """Test that resources are properly cleaned up after timeout."""
    start_threads = threading.active_count()

    @with_timeout(seconds=0.1)
    def slow_function():
        time.sleep(0.2)
        return "should not get here"

    with pytest.raises(TimeoutError):
        slow_function()

    # Allow a small amount of time for cleanup
    time.sleep(0.1)

    # Check that we haven't leaked threads
    assert threading.active_count() <= start_threads + 1  # +1 for the main thread


def test_timeout_interrupts_main_thread():
    """Test that the timeout actually interrupts the main thread."""
    @with_timeout(seconds=0.1)
    def long_running_function():
        start_time = time.time()
        while True:  # This should be interrupted
            time.sleep(0.01)
            # Add a check to break if we've been running too long
            if time.time() - start_time > 0.5:
                break

    start_time = time.time()
    with pytest.raises(TimeoutError):
        long_running_function()

    # Verify that we didn't wait too long
    assert time.time() - start_time < 0.2  # Should be interrupted well before 0.2 seconds
