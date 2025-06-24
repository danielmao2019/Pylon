from typing import Optional, Callable, Any, Dict
import threading
import time
import psutil
import torch
from concurrent.futures import ThreadPoolExecutor


class AdaptiveThreadPoolExecutor:
    """
    A ThreadPoolExecutor that dynamically adjusts worker count based on system resources.
    Drop-in replacement for ThreadPoolExecutor with adaptive scaling.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        min_workers: int = 1,
        cpu_threshold: float = 85.0,
        gpu_memory_threshold: float = 85.0,
        monitor_interval: float = 2.0,
    ):
        """
        Args:
            max_workers: Maximum number of workers (default: CPU count)
            min_workers: Minimum number of workers
            cpu_threshold: CPU usage threshold (%) above which to reduce workers
            gpu_memory_threshold: GPU memory usage threshold (%) above which to reduce workers
            monitor_interval: How often to check system resources (seconds)
        """
        self.max_workers = max_workers or psutil.cpu_count(logical=True)
        self.min_workers = max(1, min_workers)
        self.cpu_threshold = cpu_threshold
        self.gpu_memory_threshold = gpu_memory_threshold
        self.monitor_interval = monitor_interval
        
        # Start with conservative worker count
        self._current_max_workers = min(2, self.max_workers)
        self._last_adjustment_time = 0
        self._adjustment_cooldown = 3.0  # seconds
        
        self._executor = None
        self._lock = threading.Lock()
        
    def _get_system_load(self) -> Dict[str, float]:
        """Get current system resource usage."""
        # Get CPU usage over a short interval
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get GPU memory usage if available
        gpu_memory_percent = 0.0
        if torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory_total > 0:
                    gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            except:
                gpu_memory_percent = 0.0
                
        return {
            'cpu_percent': cpu_percent,
            'gpu_memory_percent': gpu_memory_percent,
        }
    
    def _adjust_workers(self) -> int:
        """Adjust worker count based on current system load."""
        current_time = time.time()
        
        # Cooldown period to prevent rapid adjustments
        if current_time - self._last_adjustment_time < self._adjustment_cooldown:
            return self._current_max_workers
            
        stats = self._get_system_load()
        
        # Check if system is under high load
        high_load = (
            stats['cpu_percent'] > self.cpu_threshold or
            stats['gpu_memory_percent'] > self.gpu_memory_threshold
        )
        
        with self._lock:
            if high_load and self._current_max_workers > self.min_workers:
                # Scale down
                self._current_max_workers = max(
                    self.min_workers,
                    self._current_max_workers - 1
                )
                self._last_adjustment_time = current_time
            elif not high_load and self._current_max_workers < self.max_workers:
                # Scale up gradually when load is low
                if (stats['cpu_percent'] < self.cpu_threshold * 0.7 and 
                    stats['gpu_memory_percent'] < self.gpu_memory_threshold * 0.7):
                    self._current_max_workers = min(
                        self.max_workers,
                        self._current_max_workers + 1
                    )
                    self._last_adjustment_time = current_time
                    
        return self._current_max_workers
    
    def get_optimal_worker_count(self) -> int:
        """Get the current optimal worker count based on system resources."""
        return self._adjust_workers()
    
    def submit(self, fn: Callable, *args, **kwargs):
        """Submit a callable to be executed with the given arguments."""
        if self._executor is None:
            optimal_workers = self.get_optimal_worker_count()
            self._executor = ThreadPoolExecutor(max_workers=optimal_workers)
        return self._executor.submit(fn, *args, **kwargs)
    
    def map(self, fn: Callable, *iterables, timeout: Optional[float] = None, chunksize: int = 1):
        """Return an iterator equivalent to map(fn, *iterables) but the calls may be evaluated out-of-order."""
        if self._executor is None:
            optimal_workers = self.get_optimal_worker_count()
            self._executor = ThreadPoolExecutor(max_workers=optimal_workers)
        return self._executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)
    
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        """Clean-up the resources associated with the Executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def create_adaptive_executor(
    max_workers: Optional[int] = None,
    min_workers: int = 1,
    **kwargs
) -> ThreadPoolExecutor:
    """
    Create an adaptive ThreadPoolExecutor that starts with optimal worker count.
    
    Args:
        max_workers: Maximum number of workers (default: CPU count)
        min_workers: Minimum number of workers
        **kwargs: Additional arguments for AdaptiveThreadPoolExecutor
        
    Returns:
        ThreadPoolExecutor configured with optimal worker count
    """
    adaptive = AdaptiveThreadPoolExecutor(
        max_workers=max_workers,
        min_workers=min_workers,
        **kwargs
    )
    
    optimal_workers = adaptive.get_optimal_worker_count()
    return ThreadPoolExecutor(max_workers=optimal_workers)