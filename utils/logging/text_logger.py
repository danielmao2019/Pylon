from typing import Any, Optional
import logging
from utils.logging.base_logger import BaseLogger


class TextLogger(BaseLogger):
    """
    A logger that writes to both a file and the console.
    Uses Python's built-in logging module.
    """
    def __init__(self, filepath: str = None) -> None:
        """
        Initialize the text logger.

        Args:
            filepath: Optional filepath to write logs to
        """
        super(TextLogger, self).__init__(filepath=filepath)
        self._init_core_logger_()

    def _init_core_logger_(self) -> None:
        """Initialize the core logger."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Create file handler if filepath is provided
        if self.filepath:
            file_handler = logging.FileHandler(self.filepath)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    async def _process_write(self, data: Any) -> None:
        """Process a write operation."""
        if isinstance(data, str):
            # Handle buffer flush
            self.logger.info(data)
        elif isinstance(data, tuple):
            msg_type, content = data
            if msg_type == "INFO":
                self.logger.info(content)
            elif msg_type == "WARNING":
                self.logger.warning(content)
            elif msg_type == "ERROR":
                self.logger.error(content)
            elif msg_type == "PAGE_BREAK":
                self.logger.info("\n" + "=" * 80 + "\n")

    async def flush(self, prefix: Optional[str] = "") -> None:
        """
        Flush the buffer to the log file and console.

        Args:
            prefix: Optional prefix to display before the data
        """
        with self._buffer_lock:
            string = prefix + ' ' + ", ".join([f"{key}: {val}" for key, val in self.buffer.items()])
            await self._write_queue.put(string)
            self.buffer = {}

    async def info(self, message: str) -> None:
        """
        Log an info message.

        Args:
            message: The message to log
        """
        await self._write_queue.put(("INFO", message))

    async def warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: The message to log
        """
        await self._write_queue.put(("WARNING", message))

    async def error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: The message to log
        """
        await self._write_queue.put(("ERROR", message))

    async def page_break(self) -> None:
        """Add a page break to the log."""
        await self._write_queue.put(("PAGE_BREAK", None))
