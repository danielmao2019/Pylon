from typing import Tuple, Optional, Any
import logging
import sys
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

    def _process_write(self, data: Tuple[str, Any]) -> None:
        """Process a write operation."""
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
        elif msg_type == "FLUSH":
            # Wait for all buffer updates to complete
            self._buffer_queue.join()
            with self._buffer_lock:
                string = content + " " + ", ".join([f"{key}: {val}" for key, val in self.buffer.items()])
                self.core_logger.info(string)
                self.buffer = {}
