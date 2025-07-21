"""Debouncing decorator for viewer callbacks to prevent excessive recomputation."""

import functools
import threading
from typing import Any, Callable


def debounce(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that delays function execution until calls have stopped for 0.5 seconds.

    This implements a simple debouncing mechanism with:
    - Fixed 0.5 second delay
    - Last-call-wins semantics (only final call executes)
    - Thread-safe implementation using Timer
    - No error handling (fail fast philosophy)

    Args:
        func: The function to debounce

    Returns:
        Debounced version of the function
    """
    # Store timer reference and lock for thread safety
    timer_lock = threading.Lock()
    current_timer = None

    @functools.wraps(func)
    def debounced(*args: Any, **kwargs: Any) -> Any:
        nonlocal current_timer

        # Cancel any existing timer
        with timer_lock:
            if current_timer is not None:
                current_timer.cancel()

            # Create new timer for this call
            current_timer = threading.Timer(0.5, lambda: func(*args, **kwargs))
            current_timer.start()

    # Store reference to timer for testing/cleanup
    debounced._timer_lock = timer_lock  # type: ignore
    debounced._get_current_timer = lambda: current_timer  # type: ignore

    return debounced
