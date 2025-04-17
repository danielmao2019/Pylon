from typing import List, Any, Optional
from abc import ABC, abstractmethod
import threading


class BaseMetric(ABC):

    def __init__(self):
        super(BaseMetric, self).__init__()
        self._buffer_lock = threading.Lock()
        self.reset_buffer()

    def reset_buffer(self):
        with self._buffer_lock:
            self.buffer: List[Any] = []

    @abstractmethod
    def __call__(self, y_pred: Any, y_true: Any) -> Any:
        raise NotImplementedError("Abstract method BaseMetric.__call__ not implemented.")

    @abstractmethod
    def summarize(self, output_path: Optional[str] = None) -> Any:
        raise NotImplementedError("Abstract method BaseMetric.summarize not implemented.")

    def add_to_buffer(self, item: Any) -> None:
        """Thread-safe method to add an item to the buffer."""
        with self._buffer_lock:
            self.buffer.append(item)

    def get_buffer(self) -> List[Any]:
        """Thread-safe method to get a copy of the buffer."""
        with self._buffer_lock:
            return self.buffer.copy()
