from typing import Optional, Generic, TypeVar
import threading
from abc import ABC, abstractmethod
from agents.connector.pool import _ssh_pool, SSHConnectionPool


T = TypeVar('T')  # Type variable for status objects


class BaseMonitor(ABC, Generic[T]):
    """Base class for single-resource monitors (CPU, GPU)."""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._monitoring_started = False
        self.ssh_pool: SSHConnectionPool = _ssh_pool
        self._initialize_state()
        try:
            self._update_resource()
        except Exception as exc:  # noqa: BLE001
            # Log error but allow monitor to continue; status remains best-effort.
            print(f"Monitor initial update failed: {exc}")

    @abstractmethod
    def _initialize_state(self) -> None:
        """Prepare monitor state (instantiate status objects, etc.)."""

    @abstractmethod
    def _update_resource(self) -> None:
        """Refresh monitor state for the underlying resource."""

    def start(self) -> None:
        if self.monitor_thread and self.monitor_thread.is_alive():
            return

        def monitor_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    self._update_resource()
                except Exception as exc:  # noqa: BLE001
                    if not self._stop_event.is_set():
                        print(f"Monitor error: {exc}")
                self._stop_event.wait(timeout=1.0)

        self._stop_event.clear()
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        self._monitoring_started = True

    def stop(self) -> None:
        self._stop_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self._monitoring_started = False

    def __del__(self) -> None:
        self.stop()

    @property
    def monitoring_active(self) -> bool:
        return self._monitoring_started
