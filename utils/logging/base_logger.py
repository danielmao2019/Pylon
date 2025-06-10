from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import os
import threading
import asyncio
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
        
        # Create log file if filepath is provided
        if self.filepath:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w') as f:
                f.write("")
        
        # Initialize async queue and worker
        self._write_queue = asyncio.Queue()
        self._write_worker = None
        self._start_write_worker()

    def _start_write_worker(self) -> None:
        """Start the async write worker."""
        self._write_worker = asyncio.create_task(self._write_worker_task())

    async def _write_worker_task(self) -> None:
        """Background task to process write operations."""
        while True:
            try:
                data = await self._write_queue.get()
                if data is None:  # Shutdown signal
                    break
                await self._process_write(data)
                self._write_queue.task_done()
            except Exception as e:
                print(f"Error in write worker: {e}")

    @abstractmethod
    async def _process_write(self, data: Any) -> None:
        """Process a write operation. Must be implemented by subclasses."""
        pass

    def update_buffer(self, data: Dict[str, Any]) -> None:
        """
        Update the buffer with new data.

        Args:
            data: Dictionary of data to update the buffer with
        """
        with self._buffer_lock:
            self.buffer.update(serialize_tensor(data))

    @abstractmethod
    async def flush(self, prefix: Optional[str] = "") -> None:
        """
        Flush the buffer to the output asynchronously.

        Args:
            prefix: Optional prefix to display before the data
        """
        pass

    @abstractmethod
    async def info(self, message: str) -> None:
        """
        Log an info message asynchronously.

        Args:
            message: The message to log
        """
        pass

    @abstractmethod
    async def warning(self, message: str) -> None:
        """
        Log a warning message asynchronously.

        Args:
            message: The message to log
        """
        pass

    @abstractmethod
    async def error(self, message: str) -> None:
        """
        Log an error message asynchronously.

        Args:
            message: The message to log
        """
        pass

    @abstractmethod
    async def page_break(self) -> None:
        """
        Add a page break to the log asynchronously.
        """
        pass

    async def shutdown(self) -> None:
        """
        Shutdown the logger gracefully.
        """
        if self._write_worker is not None:
            await self._write_queue.put(None)  # Send shutdown signal
            await self._write_worker
            self._write_worker = None
