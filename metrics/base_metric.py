from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import queue
import threading
from utils.ops.apply import apply_tensor_op


class BaseMetric(ABC):
    """
    Base class for all metrics.
    """

    def __init__(self, use_buffer: bool = True) -> None:
        """Initialize the base metric."""
        self.use_buffer = use_buffer
        if self.use_buffer:
            self._buffer_lock = threading.Lock()
            self._buffer_queue = queue.Queue()
            self._buffer_thread = threading.Thread(target=self._buffer_worker, daemon=True)
            self._buffer_thread.start()
        self.reset_buffer()

    def reset_buffer(self) -> None:
        """Reset the buffer."""
        if self.use_buffer:
            assert self._buffer_queue.empty(), "Buffer queue is not empty when resetting buffer"
            with self._buffer_lock:
                self.buffer = {}  # Now a dict mapping indices to data
        else:
            assert not hasattr(self, 'buffer')

    def _buffer_worker(self) -> None:
        """Background thread to handle buffer updates."""
        while True:
            try:
                item = self._buffer_queue.get()
                data, idx = item['data'], item['idx']
                processed_data = apply_tensor_op(func=lambda x: x.detach().cpu(), inputs=data)

                with self._buffer_lock:
                    # Store data by index for order preservation
                    self.buffer[idx] = processed_data

                self._buffer_queue.task_done()
            except Exception as e:
                print(f"Buffer worker error: {e}")

    def add_to_buffer(self, data: Dict[str, Any], idx: int) -> None:
        """
        Add data to the buffer.

        Args:
            data: Dictionary of data to add to the buffer
            idx: Integer index of the datapoint for order preservation.
        """
        if self.use_buffer:
            assert hasattr(self, 'buffer')
            assert isinstance(self.buffer, dict)
            self._buffer_queue.put({'data': data, 'idx': idx})
        else:
            assert not hasattr(self, 'buffer')

    def get_buffer(self) -> List[Any]:
        """Thread-safe method to get a copy of the buffer as an ordered list."""
        if self.use_buffer:
            with self._buffer_lock:
                # Convert dict to list, sorted by indices for order preservation
                if not self.buffer:
                    return []
                sorted_indices = sorted(self.buffer.keys())
                assert sorted_indices[0] == 0 and sorted_indices[-1] == len(self.buffer) - 1, f"{sorted_indices=}"
                return [self.buffer[idx] for idx in sorted_indices]
        raise RuntimeError("Buffer is not enabled")

    @abstractmethod
    def __call__(self, y_pred: Any, y_true: Any, idx: int) -> Any:
        raise NotImplementedError("Abstract method BaseMetric.__call__ not implemented.")

    @abstractmethod
    def summarize(self, output_path: Optional[str] = None) -> Any:
        raise NotImplementedError("Abstract method BaseMetric.summarize not implemented.")
