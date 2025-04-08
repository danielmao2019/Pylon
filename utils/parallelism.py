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
        try:
            # Acquire the semaphore
            semaphore.acquire()

            if logger:
                logger.info(f"Processing item {idx}")

            # Process the item
            result = func(*item_args)

            # Store the result
            with results_lock:
                # Ensure the results list is long enough
                while len(results) <= idx:
                    results.append(None)
                results[idx] = result

            if logger:
                logger.info(f"Completed item {idx}")

            return result
        except Exception as e:
            if logger:
                logger.error(f"Error processing item {idx}: {str(e)}")
            return None
        finally:
            # Always release the semaphore
            semaphore.release()

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

    return results
