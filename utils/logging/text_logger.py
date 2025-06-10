from typing import Optional
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

    def flush(self, prefix: Optional[str] = "") -> None:
        with self._buffer_lock:
            string = prefix + ' ' + ", ".join([f"{key}: {val}" for key, val in self.buffer.items()])
            self.core_logger.info(string)
            self.buffer = {}

    def info(self, string: str) -> None:
        self.core_logger.info(string)

    def warning(self, string: str) -> None:
        self.core_logger.warning(string)

    def error(self, string: str) -> None:
        self.core_logger.error(string)

    def page_break(self):
        self.core_logger.info("")
        self.core_logger.info('=' * 100)
        self.core_logger.info('=' * 100)
        self.core_logger.info("")
