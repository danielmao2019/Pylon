import threading
import logging
from typing import Callable, List, Any, Tuple, Optional, Iterator, Union


def parallel_process_with_semaphore(
    func: Callable,
    args: Union[List[Tuple[Any, ...]], Iterator[Tuple[Any, ...]]],
    n_jobs: int,
    logger: Optional[logging.Logger] = None,
    semaphore_timeout: int = 60,
    thread_timeout: int = 120
) -> List[Any]:
    """
    Process items in parallel using threads with a semaphore to limit concurrency.
    
    Args:
        func: The function to call for each item
        args: List or iterator of argument tuples to pass to the function
        n_jobs: Maximum number of concurrent threads
        logger: Optional logger for debugging information
        semaphore_timeout: Timeout in seconds for acquiring the semaphore
        thread_timeout: Timeout in seconds for joining threads
        
    Returns:
        List of results from the function calls
    """
    # Create a semaphore to limit concurrent processing
    semaphore = threading.Semaphore(n_jobs)
    
    # Add a counter to track active threads
    active_threads = 0
    active_threads_lock = threading.Lock()
    
    # Store results
    results = []
    results_lock = threading.Lock()
    
    # Define a function to process an item with the semaphore
    def process_item_with_semaphore(idx, item_args):
        nonlocal active_threads
        try:
            # Acquire the semaphore with a timeout to prevent deadlocks
            if not semaphore.acquire(timeout=semaphore_timeout):
                if logger:
                    logger.error(f"Timeout waiting for semaphore for item {idx}")
                return
            
            # Update active thread count
            with active_threads_lock:
                active_threads += 1
                if logger:
                    logger.info(f"Starting item {idx}, active threads: {active_threads}")
            
            # Process the item
            result = func(*item_args)
            
            # Store the result
            with results_lock:
                # Ensure the results list is long enough
                while len(results) <= idx:
                    results.append(None)
                results[idx] = result
            
            # Update active thread count
            with active_threads_lock:
                active_threads -= 1
                if logger:
                    logger.info(f"Completed item {idx}, active threads: {active_threads}")
            
            return result
        except Exception as e:
            if logger:
                logger.error(f"Error processing item {idx}: {str(e)}")
            # Make sure to release the semaphore even if there's an error
            semaphore.release()
            # Update active thread count
            with active_threads_lock:
                active_threads -= 1
                if logger:
                    logger.info(f"Error in item {idx}, active threads: {active_threads}")
    
    # Create and start threads for each item
    threads = []
    idx = 0
    
    # Process items one at a time
    for item_args in args:
        t = threading.Thread(target=process_item_with_semaphore, args=(idx, item_args))
        t.daemon = True
        t.start()
        threads.append(t)
        if logger:
            logger.info(f"Started thread for item {idx}")
        idx += 1
    
    # Wait for all threads to complete
    for i, t in enumerate(threads):
        t.join(timeout=thread_timeout)
        if t.is_alive() and logger:
            logger.error(f"Thread for item {i} did not complete within timeout")
    
    # Check if any threads are still active
    active = [t for t in threads if t.is_alive()]
    if active and logger:
        logger.error(f"{len(active)} threads are still active after join timeout")
    
    return results
