import threading
import concurrent.futures
from contextlib import contextmanager
from typing import Any


@contextmanager
def timeout(seconds: int, func_name: str):
    """A context manager that implements a timeout mechanism using threads.
    
    Args:
        seconds: Number of seconds before timeout
        func_name: Name of the function being timed out (for error message)
    
    Raises:
        TimeoutError: If the wrapped code takes longer than the specified timeout
    """
    future = concurrent.futures.Future()
    
    def target():
        try:
            # This will be set to True when the context exits
            future.set_result(None)
        except Exception as e:
            future.set_exception(e)
    
    # Create and start the thread
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    
    try:
        yield
    finally:
        # Signal that we're done
        if not future.done():
            future.set_result(None)
        
        # Wait for the result with timeout
        try:
            future.result(timeout=seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function {func_name} timed out after {seconds} seconds") 