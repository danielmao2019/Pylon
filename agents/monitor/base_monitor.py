from typing import List, Dict, Optional, Any, TypeVar, Generic
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from agents.connector.pool import _ssh_pool, SSHConnectionPool


T = TypeVar('T')  # Type variable for status objects


class BaseMonitor(ABC, Generic[T]):
    """
    Base class for system monitors (GPU, CPU) that provides common functionality
    for threading, lifecycle management, and monitoring patterns.
    """

    def __init__(self, timeout: int = 5):
        """
        Initialize base monitor with common configuration.

        Args:
            timeout: SSH command timeout in seconds
        """
        self.timeout = timeout
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.ssh_pool: SSHConnectionPool = _ssh_pool

        # Initialize status structures (implemented by subclasses)
        self._init_status_structures()
        
        # Get servers list (implemented by subclasses)
        self.servers = self._get_servers_list()

        # Do one update first
        self._update()

    @abstractmethod
    def _init_status_structures(self) -> None:
        """Initialize monitor-specific status data structures."""
        pass

    @abstractmethod
    def _get_servers_list(self) -> List[str]:
        """Get list of servers being monitored."""
        pass

    @abstractmethod
    def _update_single_server(self, server: str) -> None:
        """Update status for a single server."""
        pass

    @abstractmethod
    def _check(self) -> Dict[str, Any]:
        """Get current status of all monitored resources."""
        pass

    @abstractmethod
    def get_all_running_commands(self) -> List[str]:
        """Get all running commands on all servers."""
        pass

    def start(self):
        """Starts background monitoring thread that continuously updates info"""
        def monitor_loop():
            while not self._stop_event.is_set():
                try:
                    self._update()
                except Exception as e:
                    if not self._stop_event.is_set():
                        print(f"Monitor error: {e}")
                # Wait for a short interval or until stop event is set
                self._stop_event.wait(timeout=1.0)

        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self._stop_event.clear()
            self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
            self.monitor_thread.start()

    def stop(self):
        """Stop the monitoring thread"""
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

    def __del__(self):
        """Automatically stop monitoring when instance is destroyed"""
        self.stop()

    def _update(self):
        """Updates information for all resources using batched queries per server"""
        if self._stop_event.is_set():
            return
        try:
            with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
                list(executor.map(self._update_single_server, self.servers))
        except RuntimeError as e:
            if "cannot schedule new futures after interpreter shutdown" in str(e):
                # Interpreter is shutting down, stop the monitor
                self._stop_event.set()
            else:
                raise

    def log_stats(self, logger):
        """Logs status of all monitored resources"""
        stats = self._check()
        assert len(stats) == 1, "Only support single resource monitoring for now."
        stats = list(stats.values())[0]
        logger.update_buffer(stats)
