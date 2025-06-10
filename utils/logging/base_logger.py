from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import threading
import queue
from utils.input_checks import check_write_file
from utils.io import serialize_tensor


class BaseLogger(ABC):
    """
    Base class for loggers that provides common functionality.
    """
    def __init__(self, filepath: Optional[str] = None) -> None:
        """
        Initialize the base logger.

        Args:
            filepath: Optional filepath to write logs to
        """
        self._buffer_lock = threading.Lock()
        self.filepath = check_write_file(filepath) if filepath is not None else None
        self.buffer = {}
        self._write_queue = queue.Queue()
        self._buffer_queue = queue.Queue()
        self._write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self._buffer_thread = threading.Thread(target=self._buffer_worker, daemon=True)
        self._write_thread.start()
        self._buffer_thread.start()

        # Create log file if filepath is provided
        if self.filepath is not None:
            with open(self.filepath, 'w') as f:
                f.write("")

    def _write_worker(self) -> None:
        """Background thread to handle write operations."""
        while True:
            try:
                data = self._write_queue.get()
                if data is None:  # Shutdown signal
                    break
                self._process_write(data)
                self._write_queue.task_done()
            except Exception as e:
                print(f"Write worker error: {e}")

    def _buffer_worker(self) -> None:
        """Background thread to handle buffer updates."""
        while True:
            try:
                data = self._buffer_queue.get()
                if data is None:  # Shutdown signal
                    break
                with self._buffer_lock:
                    self.buffer.update(serialize_tensor(data))
                self._buffer_queue.task_done()
            except Exception as e:
                print(f"Buffer worker error: {e}")

    @abstractmethod
    def _process_write(self, data: Any) -> None:
        """Process a write operation."""
        pass

    def update_buffer(self, data: Dict[str, Any]) -> None:
        """
        Update the buffer with new data.

        Args:
            data: Dictionary of data to update the buffer with
        """
        self._buffer_queue.put(data)

    @abstractmethod
    def flush(self, prefix: Optional[str] = "") -> None:
        """
        Flush the buffer to the output.

        Args:
            prefix: Optional prefix to display before the data
        """
        pass

    def info(self, message: str) -> None:
        """
        Log an info message.

        Args:
            message: The message to log
        """
        self._write_queue.put(("INFO", message))

    def warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: The message to log
        """
        self._write_queue.put(("WARNING", message))

    def error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: The message to log
        """
        self._write_queue.put(("ERROR", message))

    def page_break(self) -> None:
        """Add a page break to the log."""
        self._write_queue.put(("PAGE_BREAK", None))

    def shutdown(self) -> None:
        """Shutdown the logger and wait for all messages to be processed."""
        # Wait for all buffer updates to complete
        self._buffer_queue.join()
        # Signal buffer thread to stop
        self._buffer_queue.put(None)
        # Wait for buffer thread to stop
        self._buffer_thread.join()
        # Signal write thread to stop
        self._write_queue.put(None)
        # Wait for write thread to stop
        self._write_thread.join()
