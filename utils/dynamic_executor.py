from typing import Optional, Callable, Any, Dict, List
import threading
import time
import queue
import psutil
import torch
import logging
from concurrent.futures import Future, as_completed
from dataclasses import dataclass


@dataclass
class WorkerStats:
    """Statistics for tracking worker performance."""
    worker_id: int
    tasks_completed: int = 0
    total_time: float = 0.0
    last_task_time: float = 0.0
    active: bool = True


class DynamicWorker(threading.Thread):
    """A worker thread that can be dynamically started."""

    def __init__(self, worker_id: int, task_queue: queue.Queue, result_callback: Callable):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_callback = result_callback
        self.stats = WorkerStats(worker_id)
        self._stop_event = threading.Event()

    def run(self):
        """Main worker loop."""
        while not self._stop_event.is_set():
            try:
                # Get task with timeout to allow checking stop event
                task_item = self.task_queue.get(timeout=1.0)
                if task_item is None:  # Shutdown signal
                    break

                future, func, args, kwargs = task_item
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                    logging.warning(f"Worker {self.worker_id} task failed: {e}")
                finally:
                    # Always update stats and callback regardless of success/failure
                    task_time = time.time() - start_time
                    self.stats.tasks_completed += 1
                    self.stats.total_time += task_time
                    self.stats.last_task_time = task_time

                    self.result_callback(self.worker_id, task_time)
                    self.task_queue.task_done()

            except queue.Empty:
                continue  # Check stop event and try again
            except Exception as e:
                logging.error(f"Worker {self.worker_id} error: {e}")

    def stop(self):
        """Signal the worker to stop."""
        self._stop_event.set()
        self.stats.active = False


class DynamicThreadPoolExecutor:
    """
    A truly dynamic ThreadPoolExecutor that starts with minimal workers
    and carefully scales based on resource utilization analysis.

    Key features:
    - Starts with 1 worker
    - Never scales down (workers stay once created)
    - Tracks utilization increase per worker to make informed scaling decisions
    - Uses moving averages to estimate resource impact of new workers
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        min_workers: int = 1,
        cpu_threshold: float = 85.0,
        gpu_memory_threshold: float = 85.0,
        monitor_interval: float = 2.0,
        scale_check_interval: float = 1.0,
    ):
        """
        Args:
            max_workers: Maximum number of workers (default: CPU count)
            min_workers: Minimum number of workers (and starting count)
            cpu_threshold: CPU usage threshold (%) above which to stop scaling up
            gpu_memory_threshold: GPU memory usage threshold (%) above which to stop scaling up
            monitor_interval: How often to monitor resources for scaling decisions (seconds)
            scale_check_interval: How often to check if scaling is needed (seconds)
        """
        self.max_workers = max_workers or psutil.cpu_count(logical=True)
        self.min_workers = max(1, min_workers)
        self.cpu_threshold = cpu_threshold
        self.gpu_memory_threshold = gpu_memory_threshold
        self.monitor_interval = monitor_interval
        self.scale_check_interval = scale_check_interval

        # Task management
        self.task_queue = queue.Queue()
        self.workers: List[DynamicWorker] = []
        self.next_worker_id = 0

        # Synchronization
        self._lock = threading.Lock()
        self._shutdown = False

        # Scaling state and analysis
        self._last_scale_time = 0
        self._pending_tasks = 0
        self._scale_cooldown = 5.0  # Longer cooldown for more careful scaling

        # Utilization tracking for smart scaling decisions
        self._last_resource_check = 0
        self._worker_utilization_impacts: List[Dict[str, float]] = []  # Track impact of each worker addition

        # Start with minimum workers
        self._start_initial_workers()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _start_initial_workers(self):
        """Start the initial set of workers."""
        with self._lock:
            for _ in range(self.min_workers):
                self._add_worker()

    def _add_worker(self) -> int:
        """Add a new worker. Must be called with lock held."""
        # Capture baseline utilization before adding worker
        baseline_stats = self._get_system_load()

        worker_id = self.next_worker_id
        self.next_worker_id += 1

        worker = DynamicWorker(
            worker_id=worker_id,
            task_queue=self.task_queue,
            result_callback=self._worker_completed_task
        )

        self.workers.append(worker)
        worker.start()

        # Schedule measurement of utilization impact after worker has had time to start work
        threading.Timer(2.0, self._measure_worker_impact, args=[worker_id, baseline_stats]).start()

        return worker_id

    def _worker_completed_task(self, worker_id: int, task_time: float):
        """Callback when a worker completes a task."""
        with self._lock:
            if self._pending_tasks > 0:
                self._pending_tasks -= 1

    def _get_system_load(self) -> Dict[str, float]:
        """Get current system resource usage."""
        # Get CPU usage over a short interval
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Get GPU memory usage if available
        gpu_memory_percent = 0.0
        if torch.cuda.is_available():
            try:
                # Use current device instead of hardcoded device 0
                current_device = torch.cuda.current_device()
                gpu_memory_used = torch.cuda.memory_allocated(current_device)
                gpu_memory_total = torch.cuda.get_device_properties(current_device).total_memory
                if gpu_memory_total > 0:
                    gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            except Exception as e:
                logging.debug(f"Failed to get GPU memory usage: {e}")
                gpu_memory_percent = 0.0

        return {
            'cpu_percent': cpu_percent,
            'gpu_memory_percent': gpu_memory_percent,
        }

    def _measure_worker_impact(self, worker_id: int, baseline_stats: Dict[str, float]):
        """Measure the utilization impact of adding a worker."""
        current_stats = self._get_system_load()

        # Calculate the increase in utilization
        cpu_increase = current_stats['cpu_percent'] - baseline_stats['cpu_percent']
        gpu_increase = current_stats['gpu_memory_percent'] - baseline_stats['gpu_memory_percent']

        # Only record positive increases (negative could be noise or other system activity)
        impact = {
            'worker_id': worker_id,
            'cpu_increase': max(0, cpu_increase),
            'gpu_increase': max(0, gpu_increase),
            'timestamp': time.time()
        }

        with self._lock:
            self._worker_utilization_impacts.append(impact)
            # Keep only recent measurements (last 5 workers)
            while len(self._worker_utilization_impacts) > 5:
                self._worker_utilization_impacts.pop(0)

        logging.info(f"[DynamicExecutor] Worker {worker_id} impact: CPU +{cpu_increase:.1f}%, GPU +{gpu_increase:.1f}%")

    def _estimate_worker_impact(self) -> Dict[str, float]:
        """Estimate the average resource impact of adding a new worker."""
        with self._lock:
            if not self._worker_utilization_impacts:
                # No historical data, use conservative estimates
                return {'cpu_increase': 15.0, 'gpu_increase': 10.0}

            # Calculate moving averages from recent worker additions
            recent_impacts = self._worker_utilization_impacts[-3:]  # Last 3 workers

            avg_cpu_increase = sum(impact['cpu_increase'] for impact in recent_impacts) / len(recent_impacts)
            avg_gpu_increase = sum(impact['gpu_increase'] for impact in recent_impacts) / len(recent_impacts)

            return {
                'cpu_increase': avg_cpu_increase,
                'gpu_increase': avg_gpu_increase
            }

    def _should_add_worker(self, current_stats: Dict[str, float]) -> bool:
        """
        Carefully determine if we should add more workers based on:
        1. Current utilization vs thresholds
        2. Estimated impact of adding a worker
        3. Available work in queue
        """
        with self._lock:
            current_workers = len(self.workers)

            # Don't scale beyond max
            if current_workers >= self.max_workers:
                return False

            # Must have pending work to justify more workers
            queue_has_work = not self.task_queue.empty() or self._pending_tasks > current_workers
            if not queue_has_work:
                return False

            # Estimate impact of adding another worker
            estimated_impact = self._estimate_worker_impact()

            # Check if adding a worker would exceed thresholds
            projected_cpu = current_stats['cpu_percent'] + estimated_impact['cpu_increase']
            projected_gpu = current_stats['gpu_memory_percent'] + estimated_impact['gpu_increase']

            # Be conservative - only add if we have significant headroom
            cpu_safe = projected_cpu < (self.cpu_threshold * 0.8)  # 80% of threshold
            gpu_safe = projected_gpu < (self.gpu_memory_threshold * 0.8)

            # Also check if current utilization suggests we need more workers
            # If utilization is very low, we might not need more workers even with pending tasks
            utilization_suggests_need = (
                current_stats['cpu_percent'] > 20.0 or  # Some baseline activity
                current_stats['gpu_memory_percent'] > 5.0 or
                self._pending_tasks > current_workers * 2  # Significant backlog
            )

            decision = cpu_safe and gpu_safe and utilization_suggests_need

            if decision:
                logging.debug(f"[DynamicExecutor] Planning to add worker: current={current_workers}, "
                      f"CPU {current_stats['cpu_percent']:.1f}%→{projected_cpu:.1f}%, "
                      f"GPU {current_stats['gpu_memory_percent']:.1f}%→{projected_gpu:.1f}%, "
                      f"pending={self._pending_tasks}")

            return decision

    def _monitor_loop(self):
        """Background monitoring loop for dynamic scaling."""
        while not self._shutdown:
            try:
                time.sleep(self.scale_check_interval)
                current_time = time.time()

                # Check if it's time to consider scaling
                if current_time - self._last_scale_time < self._scale_cooldown:
                    continue

                # Get current resource usage
                if current_time - self._last_resource_check >= self.monitor_interval:
                    stats = self._get_system_load()
                    self._last_resource_check = current_time

                    # Resource monitoring for scaling decisions
                    # (Historical snapshots removed as they weren't used for decision making)

                    # Consider adding workers (never scaling down)
                    if self._should_add_worker(stats):
                        with self._lock:
                            if len(self.workers) < self.max_workers:
                                worker_id = self._add_worker()
                                logging.info(f"[DynamicExecutor] Added worker {worker_id}, now {len(self.workers)} total workers")
                                self._last_scale_time = current_time

            except Exception as e:
                logging.error(f"Monitor loop error: {e}")

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a callable to be executed with the given arguments."""
        if self._shutdown:
            raise RuntimeError("Executor has been shut down")

        future = Future()
        task_item = (future, fn, args, kwargs)

        with self._lock:
            self._pending_tasks += 1
            self.task_queue.put(task_item)
        return future

    def map(self, fn: Callable, *iterables, timeout: Optional[float] = None, chunksize: int = 1):
        """Return an iterator equivalent to map(fn, *iterables)."""
        if not iterables:
            return iter([])
            
        args_list = list(zip(*iterables))
        if not args_list:
            return iter([])
            
        # Submit all tasks with their original index for order preservation
        indexed_futures = [(i, self.submit(fn, *args)) for i, args in enumerate(args_list)]
        
        def result_iterator():
            # Collect results and sort by original index to preserve order
            results = [None] * len(indexed_futures)
            completed_count = 0
            
            try:
                future_to_index = {future: i for i, future in indexed_futures}
                for future in as_completed([f for _, f in indexed_futures], timeout=timeout):
                    index = future_to_index[future]
                    results[index] = future.result()
                    completed_count += 1
                    
                # Yield results in original order
                for result in results:
                    yield result
                    
            finally:
                # Cancel any remaining futures
                for _, future in indexed_futures:
                    if not future.done():
                        future.cancel()

        return result_iterator()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        """Clean-up the resources associated with the Executor."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True

            # Signal all workers to stop
            for worker in self.workers:
                worker.stop()

            # Add None signals to wake up workers
            for _ in range(len(self.workers)):
                self.task_queue.put(None)

        if wait:
            # Wait for all workers to finish
            for worker in self.workers:
                worker.join(timeout=5.0)

            # Wait for monitor thread
            if self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @property
    def _max_workers(self) -> int:
        """Property for compatibility with ThreadPoolExecutor."""
        return self.max_workers
    
    @property
    def _current_workers(self) -> int:
        """Current number of active workers."""
        with self._lock:
            return len(self.workers)


def create_dynamic_executor(
    max_workers: Optional[int] = None,
    min_workers: int = 1,
    **kwargs
) -> DynamicThreadPoolExecutor:
    """
    Create a truly dynamic DynamicThreadPoolExecutor.

    Args:
        max_workers: Maximum number of workers (default: CPU count)
        min_workers: Minimum number of workers
        **kwargs: Additional arguments for DynamicThreadPoolExecutor

    Returns:
        DynamicThreadPoolExecutor that dynamically scales workers
    """
    return DynamicThreadPoolExecutor(
        max_workers=max_workers,
        min_workers=min_workers,
        **kwargs
    )
