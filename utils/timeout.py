from typing import TypeVar, Callable, Any
import threading
from functools import wraps

T = TypeVar('T')

def with_timeout(seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that adds a timeout to a function. If the function takes longer than
    the specified timeout, it will raise a TimeoutError.
    
    Args:
        seconds: The maximum number of seconds the function is allowed to run
        
    Returns:
        A wrapped function that will raise TimeoutError if it runs too long
        
    Example:
        @with_timeout(seconds=5)
        def long_running_function():
            time.sleep(10)  # This will raise TimeoutError after 5 seconds
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create an event to signal completion
            done = threading.Event()
            result = []
            error = []
            
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    error.append(e)
                finally:
                    done.set()
            
            # Start the function in a separate thread
            thread = threading.Thread(target=target)
            thread.daemon = True  # Make sure thread doesn't prevent program exit
            thread.start()
            
            # Wait for either completion or timeout
            if not done.wait(timeout=seconds):
                # If we timeout, we can't actually stop the thread, but we can
                # raise the error to the caller
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # If we get here, the function completed
            if error:
                raise error[0]
            return result[0]
            
        return wrapper
    return decorator
