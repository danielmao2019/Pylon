import time
import pytest
from utils.timeout import Timeout
import threading


def test_timeout_success():
    """Test that a function completing within the timeout period succeeds."""
    with Timeout(2, "test_success"):
        time.sleep(0.1)  # Should complete well within 2 seconds


def test_timeout_failure():
    """Test that a function exceeding the timeout period raises TimeoutError."""
    with pytest.raises(TimeoutError) as exc_info:
        with Timeout(0.1, "test_failure"):
            time.sleep(0.2)  # Should exceed 0.1 second timeout

    assert "test_failure" in str(exc_info.value)
    assert "timed out after 0.1 seconds" in str(exc_info.value)


def test_timeout_with_exception():
    """Test that exceptions from the wrapped code are properly propagated."""
    with pytest.raises(ValueError) as exc_info:
        with Timeout(2, "test_exception"):
            raise ValueError("Test exception")

    assert str(exc_info.value) == "Test exception"


def test_timeout_nested():
    """Test that nested timeouts work correctly."""
    with pytest.raises(TimeoutError) as exc_info:
        with Timeout(2, "outer"):
            with Timeout(0.1, "inner"):
                time.sleep(0.2)  # Should trigger inner timeout

    assert "inner" in str(exc_info.value)
    assert "timed out after 0.1 seconds" in str(exc_info.value)


def test_timeout_cleanup():
    """Test that resources are properly cleaned up after timeout."""
    start_threads = threading.active_count()

    with pytest.raises(TimeoutError):
        with Timeout(0.1, "test_cleanup"):
            time.sleep(0.2)

    # Allow a small amount of time for cleanup
    time.sleep(0.1)

    # Check that we haven't leaked threads
    assert threading.active_count() <= start_threads + 1  # +1 for the main thread
