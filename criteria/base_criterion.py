from typing import List, Any, Optional
from abc import ABC, abstractmethod
import torch
import threading
import queue


class BaseCriterion(torch.nn.Module, ABC):

    def __init__(self, use_buffer: bool = True) -> None:
        super(BaseCriterion, self).__init__()
        self.use_buffer = use_buffer
        if self.use_buffer:
            self._buffer_lock = threading.Lock()
            self._buffer_queue = queue.Queue()
            self._buffer_thread = threading.Thread(target=self._buffer_worker, daemon=True)
            self._buffer_thread.start()
        self.reset_buffer()

    def reset_buffer(self):
        if self.use_buffer:
            assert self._buffer_queue.empty(), "Buffer queue is not empty when resetting buffer"
            with self._buffer_lock:
                self.buffer: List[Any] = []
        else:
            assert not hasattr(self, 'buffer')

    def _buffer_worker(self) -> None:
        """Background thread to handle buffer updates."""
        while True:
            try:
                data = self._buffer_queue.get()
                with self._buffer_lock:
                    self.buffer.append(data.detach().cpu())
                self._buffer_queue.task_done()
            except Exception as e:
                print(f"Buffer worker error: {e}")

    def add_to_buffer(self, value: torch.Tensor) -> None:
        if self.use_buffer:
            assert hasattr(self, 'buffer')
            assert isinstance(self.buffer, list)
            assert isinstance(value, torch.Tensor), f"{type(value)=}"
            assert value.numel() == 1 and value.ndim == 0, f"{value.shape=}"
            self._buffer_queue.put(value)
        else:
            assert not hasattr(self, 'buffer')

    def get_buffer(self) -> List[Any]:
        """Thread-safe method to get a copy of the buffer."""
        if self.use_buffer:
            with self._buffer_lock:
                return self.buffer.copy()
        raise RuntimeError("Buffer is not enabled")

    @abstractmethod
    def __call__(self, y_pred: Any, y_true: Any) -> Any:
        raise NotImplementedError("Abstract method BaseCriterion.__call__ not implemented.")

    @abstractmethod
    def summarize(self, output_path: Optional[str] = None) -> Any:
        raise NotImplementedError("Abstract method BaseCriterion.summarize not implemented.")
