import threading
import time
from contextlib import contextmanager


@contextmanager
def Timeout(seconds: int, func_name: str):
    """A context manager that implements a timeout mechanism using threads.
    This is a wrapper around the contextlib.contextmanager decorator.

    Args:
        seconds: Number of seconds before timeout
        func_name: Name of the function being timed out (for error message)

    Raises:
        TimeoutError: If the wrapped code takes longer than the specified timeout
        Exception: Any exception raised by the wrapped code
    """
    start_time = time.time()
    timer = None
    timed_out = False

    def check_timeout():
        nonlocal timed_out
        if time.time() - start_time > seconds:
            timed_out = True
            return True
        return False

    def timeout_handler():
        if not check_timeout():
            # Reschedule the check
            timer = threading.Timer(0.1, timeout_handler)
            timer.start()

    try:
        # Start periodic checks
        timer = threading.Timer(0.1, timeout_handler)
        timer.start()
        try:
            yield
        except Exception as e:
            # Cancel the timer and re-raise the exception
            if timer is not None:
                timer.cancel()
            raise e
        if timed_out:
            raise TimeoutError(f"Function {func_name} timed out after {seconds} seconds")
    finally:
        if timer is not None:
            timer.cancel()
