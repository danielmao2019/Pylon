import threading
import logging
from typing import Callable, List, Any, Tuple, Optional, Iterator, Union


def parallel_execute(
    func: Callable,
    args: Union[List[Tuple[Any, ...]], Iterator[Tuple[Any, ...]]],
    n_jobs: int,
    logger: Optional[logging.Logger] = None,
    timeout: int = 30
) -> List[Optional[Any]]:
    """
    Execute a function in parallel using threads with a semaphore to limit concurrency.

    Args:
        func: The function to call for each item
        args: List or iterator of argument tuples to pass to the function
        n_jobs: Maximum number of concurrent threads
        logger: Optional logger for debugging information
        timeout: Timeout in seconds for each thread

    Returns:
        A list containing the result of each function call (often None if the function has side effects).
    """
    # Create a semaphore to limit concurrent processing
    semaphore = threading.Semaphore(n_jobs)

    # Store results
    results = []
    results_lock = threading.Lock()

    # Define a function to process an item with the semaphore
    def process_item(idx, item_args):
        item_result = None # Default result is None
        try:
            # Acquire the semaphore using 'with'
            with semaphore: # Calls semaphore.acquire() implicitly
                if logger:
                    logger.info(f"Processing item {idx}")

                # Process the item
                item_result = func(*item_args)

                if logger:
                    logger.info(f"Completed item {idx}")
            # semaphore.release() is called implicitly here when exiting 'with' block

        except Exception as e:
            if logger:
                logger.error(f"Error processing item {idx}: {str(e)}")
            # If an error occurs *inside* the 'with semaphore:', release() was still called.
            item_result = e # Store exception as result for errors
        finally:
            # Store the result (None, the function's return, or an Exception)
            # This is outside the 'with semaphore' but still needed to record the outcome.
            with results_lock:
                # Ensure the results list is long enough
                while len(results) <= idx:
                    results.append(None)
                results[idx] = item_result
            # The finally block is no longer needed *just* for semaphore release.

    # Create and start threads for each item
    threads = []
    idx = 0

    # Process items one at a time
    for item_args in args:
        t = threading.Thread(target=process_item, args=(idx, item_args))
        t.daemon = True
        t.start()
        threads.append(t)
        idx += 1

    # Wait for all threads to complete
    for i, t in enumerate(threads):
        t.join(timeout=timeout)
        if t.is_alive() and logger:
            logger.warning(f"Thread for item {i} did not complete within timeout")

    # Ensure the results list has the correct size if threads timed out / errored early
    with results_lock:
        while len(results) < idx:
             results.append(None) # Append None for items that didn't finish

    return results
