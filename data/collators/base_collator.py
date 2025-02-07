from typing import List, Dict, Callable, Union, Any, Optional
import torch
from utils.ops import transpose_buffer


class BaseCollator:

    def __init__(self, collators: Optional[Dict[str, Dict[str, Callable[[List[Any]], Any]]]] = None) -> None:
        """
        Initialize the BaseCollator with optional custom collators.

        Args:
            collators: A dictionary specifying custom collate functions for
                       specific keys. The structure is `Dict[key1, Dict[key2, Callable]]`.
        """
        self.collators = collators or {}

    def __call__(self, datapoints: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Processes a batch of datapoints using the defined or default collation logic.

        Args:
            datapoints: A list of nested dictionaries to be processed.

        Returns:
            A dictionary with collated data.
        """
        # Transpose the first level of the datapoints buffer
        datapoints = transpose_buffer(datapoints)

        for key1, sub_dict in datapoints.items():
            # Transpose the second level
            datapoints[key1] = transpose_buffer(sub_dict)

            for key2, values in datapoints[key1].items():
                # Check for custom collator
                if key1 in self.collators and key2 in self.collators[key1]:
                    datapoints[key1][key2] = self.collators[key1][key2](values)
                else:
                    # Default collation behavior
                    datapoints[key1][key2] = self._default_collate(values, key1, key2)

        return datapoints

    @staticmethod
    def _default_collate(values: List[Any], key1: str, key2: str) -> Union[torch.Tensor, List[Any]]:
        """
        Default collation logic for handling common types.

        Args:
            values: A list of values to be collated.
            key1: Outer key for context in error messages.
            key2: Inner key for context in error messages.

        Returns:
            A collated tensor or the original list if collation is not possible.
        """
        if all(value is None or isinstance(value, str) for value in values):
            # Leave strings and None values as-is
            return values

        if all(isinstance(value, int) for value in values):
            # Convert integers to a tensor
            return torch.tensor(values, dtype=torch.int64)

        try:
            # Default behavior: stack tensors
            return torch.stack(values, dim=0)
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Cannot stack tensors into a batch for key1='{key1}', key2='{key2}'. "
                f"Details: {e}"
            )
