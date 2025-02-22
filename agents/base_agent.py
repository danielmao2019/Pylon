from typing import List
from abc import ABC
import os


class BaseAgent(ABC):

    def __init__(
        self,
        config_files: List[str],
        expected_files: List[str],
    ) -> None:
        self.config_files = config_files
        self.expected_files = expected_files

    @staticmethod
    def _get_work_dir(config_file: str) -> str:
        return os.path.splitext(os.path.join("./logs", os.path.relpath(config_file, start="./configs")))[0]
