from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import threading
import queue
from utils.io.json import serialize_tensor


class BaseMetric(ABC):
    """
    Base class for all metrics.
    """

    def __init__(self) -> None:
        """Initialize the base metric."""
        self._buffer_lock = threading.Lock()
        self._buffer_queue = queue.Queue()
        self._buffer_thread = threading.Thread(target=self._buffer_worker, daemon=True)
        self.reset_buffer()
        self._buffer_thread.start()

    def reset_buffer(self) -> None:
        """Reset the buffer."""
        assert self._buffer_queue.empty(), "Buffer queue is not empty when resetting buffer"
        assert not self._buffer_thread.is_alive(), "Buffer thread is still running when resetting buffer"
        with self._buffer_lock:
            self.buffer = []

    def _buffer_worker(self) -> None:
        """Background thread to handle buffer updates."""
        while True:
            try:
                data = self._buffer_queue.get()
                with self._buffer_lock:
                    self.buffer.append(serialize_tensor(data))
                self._buffer_queue.task_done()
            except Exception as e:
                print(f"Buffer worker error: {e}")

    def add_to_buffer(self, data: Dict[str, Any]) -> None:
        """
        Add data to the buffer.

        Args:
            data: Dictionary of data to add to the buffer
        """
        self._buffer_queue.put(data)

    def get_buffer(self) -> List[Any]:
        """Thread-safe method to get a copy of the buffer."""
        with self._buffer_lock:
            return self.buffer.copy()

    @abstractmethod
    def __call__(self, y_pred: Any, y_true: Any) -> Any:
        raise NotImplementedError("Abstract method BaseMetric.__call__ not implemented.")

    @abstractmethod
    def summarize(self, output_path: Optional[str] = None) -> Any:
        raise NotImplementedError("Abstract method BaseMetric.summarize not implemented.")
