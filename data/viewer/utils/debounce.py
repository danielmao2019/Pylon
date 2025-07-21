"""Debouncing decorator for viewer callbacks to prevent excessive recomputation."""

import functools
import threading
import time
from typing import Any, Callable, Dict
from dash.exceptions import PreventUpdate

import logging
logger = logging.getLogger(__name__)

# Global state for tracking last execution results (reserved for future use)
# _debounce_cache: Dict[str, Any] = {}
# _debounce_lock = threading.Lock()


def debounce(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that implements "last call wins" debouncing for Dash callbacks.

    This implements a debouncing mechanism that:
    - Executes first call immediately to show immediate feedback
    - Schedules last call to execute after 1 second delay
    - Cancels intermediate calls but preserves the final user interaction
    - Uses PreventUpdate for intermediate calls to avoid UI updates

    Args:
        func: The function to debounce

    Returns:
        Debounced version of the function
    """
    # Store timer reference and state for this specific function
    timer_lock = threading.Lock()
    current_timer = None
    last_execution_time = 0
    last_result = None
    has_executed_first = False

    @functools.wraps(func)
    def debounced(*args: Any, **kwargs: Any) -> Any:
        nonlocal current_timer, last_execution_time, last_result, has_executed_first
        
        current_time = time.time()
        time_since_last = current_time - last_execution_time

        with timer_lock:
            # First call ever - execute immediately for immediate feedback
            if not has_executed_first:
                has_executed_first = True
                last_execution_time = current_time
                last_result = func(*args, **kwargs)
                return last_result
            
            # If enough time has passed since last execution, execute immediately
            if time_since_last >= 1.0:
                last_execution_time = current_time
                last_result = func(*args, **kwargs)
                return last_result
            
            # Rapid call - schedule delayed execution of this call (last call wins)
            if current_timer is not None:
                current_timer.cancel()
            
            def delayed_execution():
                nonlocal last_execution_time, last_result
                with timer_lock:
                    last_execution_time = time.time()
                    last_result = func(*args, **kwargs)
            
            # Schedule execution after remaining time to reach 1 second delay
            remaining_delay = 1.0 - time_since_last
            current_timer = threading.Timer(remaining_delay, delayed_execution)
            current_timer.start()
            
            # Prevent this intermediate call from updating UI
            raise PreventUpdate

    # Store reference to timer for testing/cleanup
    debounced._timer_lock = timer_lock  # type: ignore
    debounced._get_current_timer = lambda: current_timer  # type: ignore

    return debounced
