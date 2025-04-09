from typing import Callable, List, Any, Tuple, Optional, Iterator, Union
import logging
import threading # Add threading back for the helper function
import torch.multiprocessing as mp


def parallel_execute(
    func: Callable,
    args: Union[List[Tuple[Any, ...]], Iterator[Tuple[Any, ...]]],
    n_jobs: int,
    logger: Optional[logging.Logger] = None,
    timeout: int = 30,
    parallelization_type: str = 'threading' # Default to threading
) -> List[Optional[Any]]:
    """
    Execute a function in parallel using either threads or processes.

    Args:
        func: The function to call for each item.
        args: List or iterator of argument tuples to pass to the function.
        n_jobs: Maximum number of concurrent workers (threads or processes).
        logger: Optional logger for status/errors.
        timeout: Timeout in seconds for each worker (primarily effective for 'threading').
        parallelization_type: 'threading' or 'multiprocessing'. Specifies the parallel backend.

    Returns:
        A list containing the result of each function call (often None or Exception objects).
        For multiprocessing, results from workers might need manual aggregation by the caller
        if the function has side effects (like updating metrics).

    Raises:
        ValueError: If an invalid parallelization_type is provided.
    """
    if parallelization_type == 'threading':
        return _parallel_execute_threading(func, args, n_jobs, logger, timeout)
    elif parallelization_type == 'multiprocessing':
        return _parallel_execute_multiprocessing(func, args, n_jobs, logger)
    else:
        raise ValueError(f"Invalid parallelization_type: '{parallelization_type}'. Must be 'threading' or 'multiprocessing'.")


def _parallel_execute_threading(
    func: Callable,
    args: Union[List[Tuple[Any, ...]], Iterator[Tuple[Any, ...]]],
    n_jobs: int,
    logger: Optional[logging.Logger] = None,
    timeout: int = 30
) -> List[Optional[Any]]:
    semaphore = threading.Semaphore(n_jobs)
    results = []
    results_lock = threading.Lock()

    def process_item(idx, item_args):
        item_result = None
        try:
            with semaphore:
                item_result = func(*item_args)
        except Exception as e:
            if logger:
                logger.error(f"[Thread] Error processing item {idx}: {e}", exc_info=True)
            item_result = e
        finally:
            with results_lock:
                while len(results) <= idx:
                    results.append(None)
                results[idx] = item_result

    threads = {}
    idx = 0
    for item_args in args:
        t = threading.Thread(target=process_item, args=(idx, item_args))
        t.daemon = True
        t.start()
        threads[idx] = t
        idx += 1

    for i, t in threads.items():
        t.join(timeout=timeout)
        if t.is_alive() and logger:
            logger.warning(f"[Thread] Thread for item {i} did not complete within timeout")

    with results_lock:
        while len(results) < idx:
             results.append(None)

    return results


def _parallel_execute_multiprocessing(
    func: Callable,
    args: Union[List[Tuple[Any, ...]], Iterator[Tuple[Any, ...]]],
    n_jobs: int,
    logger: Optional[logging.Logger] = None,
) -> List[Optional[Any]]:
    """
    NOTE: Requires func and args items to be picklable.
    NOTE: Does not support per-task timeout easily.
    NOTE: State changes within func (e.g., to self.metric) will NOT affect the parent process.
          Results must be collected and aggregated manually by the caller.
    """
    ctx = mp.get_context('spawn')
    results = []
    try:
        args_list = list(args) if isinstance(args, Iterator) else args

        if not args_list:
             return []

        with ctx.Pool(processes=n_jobs) as pool:
            results = pool.starmap(func, args_list)
            if logger:
                 logger.info(f"[Multiprocessing] Pool completed {len(results)} tasks.")

    except Exception as e:
        if logger:
            logger.error(f"[Multiprocessing] Pool encountered an error: {e}", exc_info=True)
        results = [] # Return empty on error

    return results
