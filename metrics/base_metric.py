from typing import List, Any, Optional, Union
from abc import ABC, abstractmethod
import threading
import multiprocessing as mp
import os


class BaseMetric(ABC):

    def __init__(self, shared_buffer=None):
        super(BaseMetric, self).__init__()
        # Initialize without a lock - we'll create it when needed
        self._lock = None
        
        # Support for shared buffer between processes
        if shared_buffer is not None:
            # Use the provided shared buffer
            self._shared_buffer = shared_buffer
            self._is_shared = True
        else:
            # Create a local buffer
            self._shared_buffer = None
            self._is_shared = False
            
        self.reset_buffer()

    def _get_lock(self):
        """Get or create a lock appropriate for the current context."""
        if self._lock is None:
            # Create a lock based on whether we're in the main thread or a worker process
            if threading.current_thread().name == 'MainThread':
                self._lock = threading.Lock()
            else:
                self._lock = mp.Lock()
        return self._lock

    def reset_buffer(self) -> None:
        """Reset the buffer to empty state."""
        if self._shared_buffer is not None:
            # Clear the shared buffer by replacing all elements with an empty list
            self._shared_buffer[:] = []
        else:
            self.buffer = []

    @abstractmethod
    def __call__(self, y_pred: Any, y_true: Any) -> Any:
        raise NotImplementedError("Abstract method BaseMetric.__call__ not implemented.")

    @abstractmethod
    def summarize(self, output_path: Optional[str] = None) -> Any:
        raise NotImplementedError("Abstract method BaseMetric.summarize not implemented.")

    def get_buffer(self) -> List[Any]:
        """Get a copy of the buffer."""
        with self._get_lock():
            if self._is_shared:
                # Return a copy of the shared buffer
                return list(self._shared_buffer)
            else:
                # Return a copy of the local buffer
                return self.buffer.copy()

    def add_to_buffer(self, item: Any) -> None:
        """Thread/process-safe method to add an item to the buffer."""
        with self._get_lock():
            if self._is_shared:
                # Add to the shared buffer
                self._shared_buffer.append(item)
            else:
                # Add to the local buffer
                self.buffer.append(item)

    def set_shared_buffer(self, shared_buffer) -> None:
        """Set a shared buffer for inter-process communication."""
        with self._get_lock():
            self._shared_buffer = shared_buffer
            self._is_shared = True
