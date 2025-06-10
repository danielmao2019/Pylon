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
        self._write_queue = asyncio.Queue()
        self._write_worker = asyncio.create_task(self._write_worker_task())

        # Create log file if filepath is provided
        if self.filepath:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w') as f:
                f.write("")

    async def _write_worker_task(self) -> None:
        """Background task to process write operations."""
        while True:
            data = await self._write_queue.get()
            await self._process_write(data)
            self._write_queue.task_done()

    @abstractmethod
    async def _process_write(self, data: Any) -> None:
        """Process a write operation."""
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
        Flush the buffer to the output.

        Args:
            prefix: Optional prefix to display before the data
        """
        pass

    @abstractmethod
    async def info(self, message: str) -> None:
        """
        Log an info message.

        Args:
            message: The message to log
        """
        pass

    @abstractmethod
    async def warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: The message to log
        """
        pass

    @abstractmethod
    async def error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: The message to log
        """
        pass

    @abstractmethod
    async def page_break(self) -> None:
        """
        Add a page break to the log.
        """
        pass

    async def shutdown(self) -> None:
        """Shutdown the logger and wait for all messages to be processed."""
        await self._write_queue.join()
        self._write_worker.cancel()
        try:
            await self._write_worker
        except asyncio.CancelledError:
            pass
