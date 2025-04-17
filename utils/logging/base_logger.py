from typing import Dict, Any, Optional
import os
import threading
from abc import ABC, abstractmethod

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

    def update_buffer(self, data: Dict[str, Any]) -> None:
        """
        Update the buffer with new data.

        Args:
            data: Dictionary of data to update the buffer with
        """
        with self._buffer_lock:
            self.buffer.update(serialize_tensor(data))

    @abstractmethod
    def flush(self, prefix: Optional[str] = "") -> None:
        """
        Flush the buffer to the output.

        Args:
            prefix: Optional prefix to display before the data
        """
        pass

    @abstractmethod
    def info(self, message: str) -> None:
        """
        Log an info message.

        Args:
            message: The message to log
        """
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: The message to log
        """
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: The message to log
        """
        pass

    @abstractmethod
    def page_break(self) -> None:
        """
        Add a page break to the log.
        """
        pass
