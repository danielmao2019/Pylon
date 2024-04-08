import torch
import logging
import sys

from ..ops import apply_tensor_op


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

    def __init__(self, filepath: str):
        self._init_logger_(filepath=filepath)
        self._init_file_handler_()
        self._init_stream_handler_()

    def _init_logger_(self, filepath: str):
        assert type(filepath) == str, f"{type(filepath)=}"
        self._filepath_ = filepath
        self._logger_ = logging.getLogger(name=self._filepath_)
        self._logger_.setLevel(level=logging.INFO)
        self._info_buffer_ = {}

    def _init_file_handler_(self):
        f_handler = logging.FileHandler(filename=self._filepath_)
        f_handler.setFormatter(self.formatter)
        f_handler.setLevel(level=logging.INFO)
        self._logger_.addHandler(f_handler)

    def _init_stream_handler_(self):
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(self.formatter)
        s_handler.setLevel(level=logging.INFO)
        self._logger_.addHandler(s_handler)

    ####################################################################################################
    ####################################################################################################

    def update_buffer(self, data: dict):
        def serialize(value):
            if type(value) == torch.Tensor:
                value = value.detach()
                if value.numel() > 1:
                    value = value.tolist()
                else:
                    value = value.item()
            return value
        data = apply_tensor_op(func=serialize, inputs=data)
        self._info_buffer_.update(data)

    def flush(self, prefix: str = ""):
        string = prefix + ' ' + ", ".join([f"{key}: {val}" for key, val in self._info_buffer_.items()])
        self._logger_.info(string)
        self._info_buffer_ = {}

    def info(self, string):
        self._logger_.info(string)

    def warning(self, string):
        self._logger_.warning(string)

    def error(self, string):
        self._logger_.error(string)

    ####################################################################################################
    ####################################################################################################

    def page_break(self):
        self._logger_.info("")
        self._logger_.info('=' * 100)
        self._logger_.info('=' * 100)
        self._logger_.info("")
