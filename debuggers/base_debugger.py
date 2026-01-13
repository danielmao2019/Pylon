from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class BaseDebugger(ABC):
    """Base class for all debuggers."""

    @abstractmethod
    def __call__(self, datapoint: Dict[str, Any], model: torch.nn.Module) -> Any:
        """Process datapoint and return debug output.

        Args:
            datapoint: Dict with inputs, labels, meta_info, outputs
            model: The model being debugged

        Returns:
            Debug output in any format (dict, tensor, list, etc.)
        """
        raise NotImplementedError
