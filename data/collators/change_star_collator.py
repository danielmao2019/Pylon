from typing import List, Dict, Any, Optional
import random
import torch
from data.collators import BaseCollator
from utils.ops import transpose_buffer


class ChangeStarCollator(BaseCollator):
    __doc__ = r"""A collator for the ChangeStar algorithm, handling image pairs and their corresponding change maps.

    Reference:
        * https://github.com/Z-Zheng/ChangeStar/blob/master/core/mixin.py
    """

    METHOD_OPTIONS = ['train', 'eval']

    def __init__(self, method: str, max_trails: Optional[int] = 50) -> None:
        r"""
        Args:
            max_trails (Optional[int]): The maximum number of attempts to shuffle inputs without collisions.
        """
        super(ChangeStarCollator, self).__init__()
        assert method in self.METHOD_OPTIONS, f"{method=}"
        self.method = method
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
        if self.method == 'train':
            return self._call_train(datapoints)
        else:
            return self._call_eval(datapoints)

    def _call_train(self, datapoints: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Process a batch of datapoints to create image pairs and their corresponding change maps.

        Args:
            datapoints (List[Dict[str, Dict[str, Any]]]): The input data.

        Returns:
            Dict[str, Dict[str, Any]]: The processed batch.
        """
        batch_size = len(datapoints)
        datapoints = transpose_buffer(datapoints)
        original_indices = list(range(batch_size))
        shuffled_indices = self._shuffle(original_indices, self.max_trails)

        # Process inputs
        datapoints["inputs"] = transpose_buffer(datapoints["inputs"])
        assert set(datapoints["inputs"].keys()) == set(['img_1', 'img_2'])
        datapoints["inputs"]["img_1"] = torch.stack(datapoints["inputs"]["img_1"], dim=0)
        del datapoints["inputs"]["img_2"]
        datapoints["inputs"]["img_2"] = datapoints["inputs"]["img_1"][shuffled_indices]
        assert set(datapoints['inputs'].keys()) == set(['img_1', 'img_2'])

        # Process labels
        datapoints["labels"] = transpose_buffer(datapoints["labels"])
        assert set(datapoints['labels'].keys()) == set(['lbl_1', 'lbl_2'])
        datapoints["labels"]["semantic"] = torch.stack(datapoints["labels"]["lbl_1"], dim=0)
        del datapoints["labels"]["lbl_1"], datapoints["labels"]["lbl_2"]
        datapoints["labels"]["change"] = (datapoints["labels"]["semantic"] != datapoints["labels"]["semantic"][shuffled_indices]).to(torch.int64)
        assert set(datapoints['labels'].keys()) == set(['change', 'semantic'])

        # Process meta information
        datapoints["meta_info"] = transpose_buffer(datapoints["meta_info"])
        for key, values in datapoints["meta_info"].items():
            datapoints["meta_info"][key] = self._default_collate(
                values=values, key1="meta_info", key2=key,
            )

        return datapoints

    def _call_eval(self, datapoints: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
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
        datapoints["inputs"] = transpose_buffer(datapoints["inputs"])
        assert set(datapoints["inputs"].keys()) == set(['img_1', 'img_2'])
        datapoints["inputs"]["img_1"] = torch.stack(datapoints["inputs"]["img_1"], dim=0)
        datapoints["inputs"]["img_2"] = torch.stack(datapoints["inputs"]["img_2"], dim=0)
        assert set(datapoints["inputs"].keys()) == set(['img_1', 'img_2'])

        # Process labels
        datapoints["labels"] = transpose_buffer(datapoints["labels"])
        assert set(datapoints['labels'].keys()) == set(['lbl_1', 'lbl_2'])
        datapoints["labels"]["semantic_1"] = torch.stack(datapoints["labels"]["lbl_1"], dim=0)
        datapoints["labels"]["semantic_2"] = torch.stack(datapoints["labels"]["lbl_2"], dim=0)
        del datapoints["labels"]["lbl_1"], datapoints["labels"]["lbl_2"]
        datapoints["labels"]["change"] = (datapoints["labels"]["semantic_1"] != datapoints["labels"]["semantic_2"]).to(torch.int64)
        assert set(datapoints['labels'].keys()) == set(['change', 'semantic_1', 'semantic_2'])

        # Process meta information
        datapoints["meta_info"] = transpose_buffer(datapoints["meta_info"])
        for key, values in datapoints["meta_info"].items():
            datapoints["meta_info"][key] = self._default_collate(
                values=values, key1="meta_info", key2=key,
            )

        return datapoints

    def _call_eval(self, datapoints: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
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
        datapoints["inputs"] = transpose_buffer(datapoints["inputs"])
        assert set(datapoints["inputs"].keys()) == set(['img_1', 'img_2'])
        datapoints["inputs"]["img_1"] = torch.stack(datapoints["inputs"]["img_1"], dim=0)
        del datapoints["inputs"]["img_2"]

        # Process labels
        datapoints["labels"] = transpose_buffer(datapoints["labels"])
        assert set(datapoints['labels'].keys()) == set(['lbl_1', 'lbl_2'])
        datapoints["labels"]["semantic_1"] = torch.stack(datapoints["labels"]["lbl_1"], dim=0)
        datapoints["labels"]["semantic_2"] = torch.stack(datapoints["labels"]["lbl_2"], dim=0)
        del datapoints["labels"]["lbl_1"], datapoints["labels"]["lbl_2"]

        # Shuffle and compute change map
        original_indices = list(range(batch_size))
        shuffled_indices = self._shuffle(original_indices, self.max_trails)
        datapoints["inputs"]["img_2"] = datapoints["inputs"]["img_1"][shuffled_indices]
        datapoints["labels"]["change"] = (datapoints["labels"]["semantic_1"] != datapoints["labels"]["semantic_1"][shuffled_indices]).to(torch.int64)

        # Process meta information
        datapoints["meta_info"] = transpose_buffer(datapoints["meta_info"])
        for key, values in datapoints["meta_info"].items():
            datapoints["meta_info"][key] = self._default_collate(
                values=values, key1="meta_info", key2=key,
            )

        # Output check
        assert set(datapoints['inputs'].keys()) == set(['img_1', 'img_2'])
        assert set(datapoints['labels'].keys()) == set(['change', 'semantic_1', 'semantic_2'])

        return datapoints
