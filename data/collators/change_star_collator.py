from typing import List, Dict, Any, Optional
import random
import torch
from data.collators import BaseCollator
from utils.ops import transpose_buffer


class ChangeStarCollator(BaseCollator):
    """
    A collator for ChangeStar models, handling image pairs and their corresponding change maps.

    Args:
        max_trails (Optional[int]): The maximum number of attempts to shuffle inputs without collisions.
    """

    def __init__(self, max_trails: Optional[int] = 50) -> None:
        super().__init__()
        if max_trails is not None:
            if not isinstance(max_trails, int) or max_trails <= 0:
                raise ValueError(f"`max_trails` must be a positive integer. Got: {max_trails}")
        self.max_trails = max_trails

    @staticmethod
    def _shuffle(original: List[int], max_trails: int) -> List[int]:
        """
        Shuffle a list while ensuring no element remains in its original position.

        Args:
            original (List[int]): The original list to shuffle.
            max_trails (int): Maximum number of shuffling attempts.

        Returns:
            List[int]: The shuffled list.
        """
        for _ in range(max_trails):
            proposal = original.copy()
            random.shuffle(proposal)
            if all(x != y for x, y in zip(original, proposal)):
                return proposal
        raise RuntimeError("Failed to shuffle without collisions after max_trails attempts.")

    def __call__(self, datapoints: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Process a batch of datapoints to create image pairs and their corresponding change maps.

        Args:
            datapoints (List[Dict[str, Dict[str, Any]]]): The input data.

        Returns:
            Dict[str, Dict[str, Any]]: The processed batch.
        """
        batch_size = len(datapoints)
        datapoints = transpose_buffer(datapoints)

        # Process inputs
        if "inputs" not in datapoints or "image" not in datapoints["inputs"]:
            raise KeyError("Expected `datapoints['inputs']['image']` to be present.")
        datapoints["inputs"] = transpose_buffer(datapoints["inputs"])
        datapoints["inputs"]["img_1"] = torch.stack(datapoints["inputs"]["image"], dim=0)
        del datapoints["inputs"]["image"]

        # Process labels
        if "labels" not in datapoints or "semantic_segmentation" not in datapoints["labels"]:
            raise KeyError("Expected `datapoints['labels']['semantic_segmentation']` to be present.")
        datapoints["labels"] = transpose_buffer(datapoints["labels"])
        lbl_1 = torch.stack(datapoints["labels"]["semantic_segmentation"], dim=0)
        del datapoints["labels"]["semantic_segmentation"]

        # Shuffle and compute change map
        original_indices = list(range(batch_size))
        shuffled_indices = self._shuffle(original_indices, self.max_trails)
        datapoints["inputs"]["img_2"] = datapoints["inputs"]["img_1"][shuffled_indices]
        datapoints["labels"]["change_map"] = torch.logical_xor(lbl_1, lbl_1[shuffled_indices])

        # Process meta information
        if "meta_info" in datapoints:
            for key, values in datapoints["meta_info"].items():
                datapoints["meta_info"][key] = self._default_collate(
                    values=values, key1="meta_info", key2=key
                )

        return datapoints
