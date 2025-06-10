from typing import Any, Optional
import logging
import sys
import threading

from utils.input_checks import check_write_file
from utils.io import serialize_tensor
from utils.logging.base_logger import BaseLogger


class TextLogger(BaseLogger):
    """
    A text-based logger that writes to both a file and the console.
    """

    formatter = logging.Formatter(
        fmt=f"[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # A more complete version of formatter
    # formatter = logging.Formatter(
    #     fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) %(pathname)s F%(funcName)s L%(lineno)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )

    # ====================================================================================================
    # init methods
    # ====================================================================================================

    def __init__(self, filepath: Optional[str] = None) -> None:
        super(TextLogger, self).__init__(filepath=filepath)
        self._init_core_logger_()
        if not self.core_logger.handlers:
            self._init_file_handler_()
            self._init_stream_handler_()

    def _init_core_logger_(self) -> None:
        self.core_logger = logging.getLogger(name=self.filepath)
        self.core_logger.setLevel(level=logging.INFO)
        self.core_logger.propagate = True

    def _init_file_handler_(self) -> None:
        if self.filepath is None:
            return
        f_handler = logging.FileHandler(filename=self.filepath)
        f_handler.setFormatter(self.formatter)
        f_handler.setLevel(level=logging.INFO)
        self.core_logger.addHandler(f_handler)

    def _init_stream_handler_(self) -> None:
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(self.formatter)
        s_handler.setLevel(level=logging.INFO)
        self.core_logger.addHandler(s_handler)

    # ====================================================================================================
    # logging methods
    # ====================================================================================================

    async def _process_write(self, data: Any) -> None:
        """Process a write operation."""
        if isinstance(data, str):
            self.core_logger.info(data)
        elif isinstance(data, tuple):
            msg_type, content = data
            if msg_type == "INFO":
                self.core_logger.info(content)
            elif msg_type == "WARNING":
                self.core_logger.warning(content)
            elif msg_type == "ERROR":
                self.core_logger.error(content)
            elif msg_type == "PAGE_BREAK":
                self.core_logger.info("")
                self.core_logger.info('=' * 100)
                self.core_logger.info('=' * 100)
                self.core_logger.info("")

    async def flush(self, prefix: Optional[str] = "") -> None:
        with self._buffer_lock:
            string = prefix + ' ' + ", ".join([f"{key}: {val}" for key, val in self.buffer.items()])
            await self._write_queue.put(string)
            self.buffer = {}

    async def info(self, string: str) -> None:
        await self._write_queue.put(("INFO", string))

    async def warning(self, string: str) -> None:
        await self._write_queue.put(("WARNING", string))

    async def error(self, string: str) -> None:
        await self._write_queue.put(("ERROR", string))

    async def page_break(self) -> None:
        await self._write_queue.put(("PAGE_BREAK", None))
