from typing import Optional
import logging
import sys

from ..input_checks import check_write_file
from ..io import serialize_tensor


class Logger:

    formatter = logging.Formatter(
        fmt=f"[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # A more complete version of formatter
    # formatter = logging.Formatter(
    #     fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) %(pathname)s F%(funcName)s L%(lineno)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )

    def __init__(self, filepath: Optional[str] = None) -> None:
        self._init_core_logger_(filepath=filepath)
        self._init_file_handler_()
        self._init_stream_handler_()

    def _init_core_logger_(self, filepath: Optional[str]) -> None:
        self.filepath = check_write_file(filepath) if filepath is not None else None
        self.core_logger = logging.getLogger(name=self.filepath)
        self.core_logger.setLevel(level=logging.INFO)
        self.buffer = {}

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

    ####################################################################################################
    ####################################################################################################

    def update_buffer(self, data: dict) -> None:
        self.buffer.update(serialize_tensor(data))

    def flush(self, prefix: Optional[str] = "") -> None:
        string = prefix + ' ' + ", ".join([f"{key}: {val}" for key, val in self.buffer.items()])
        self.core_logger.info(string)
        self.buffer = {}

    def info(self, string: str) -> None:
        self.core_logger.info(string)

    def warning(self, string: str) -> None:
        self.core_logger.warning(string)

    def error(self, string: str) -> None:
        self.core_logger.error(string)

    ####################################################################################################
    ####################################################################################################

    def page_break(self):
        self.core_logger.info("")
        self.core_logger.info('=' * 100)
        self.core_logger.info('=' * 100)
        self.core_logger.info("")
