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
