from typing import Tuple, List, Dict, Any
from abc import ABC, abstractmethod
import os
import torch
from transforms import BaseTransform
from utils.builder import build_from_config


class BaseDataset(torch.utils.data.Dataset, ABC):

    SPLIT_OPTIONS: List[str] = None
    INPUT_NAMES: List[str] = None
    LABEL_NAMES: List[str] = None
    image_filepaths: List[str] = None

    def __init__(
        self,
        data_root: str,
        split: str,
        transforms: dict,
        indices: List[int] = None,
    ):
        super(BaseDataset, self).__init__()
        assert self.SPLIT_OPTIONS is not None
        assert self.INPUT_NAMES is not None
        assert self.LABEL_NAMES is not None
        assert set(self.INPUT_NAMES) & set(self.LABEL_NAMES) == set(), f"{set(self.INPUT_NAMES) & set(self.LABEL_NAMES)=}"
        # initialize data root directory
        assert type(data_root) == str, f"{type(data_root)=}"
        assert os.path.isdir(data_root), f"{data_root=}"
        self.data_root = data_root
        # init file paths
        self._init_images_(split=split)
        self._init_labels_(split=split)
        # init transforms
        self._init_transform_(transforms)
        self.indices = indices

    @abstractmethod
    def _init_images_(self, split: str) -> None:
        raise NotImplementedError("[ERROR] _init_images_ not implemented for abstract base class.")

    @abstractmethod
    def _init_labels_(self, split: str) -> None:
        raise NotImplementedError("[ERROR] _init_labels_ not implemented for abstract base class.")

    def _init_transform_(self, transforms: dict):
        if transforms is None:
            transforms = {
                'class': BaseTransform,
                'args': {
                    'transforms': {},
                },
            }
        self.transforms = build_from_config(transforms)

    def __len__(self):
        if self.indices is None:
            return len(self.image_filepaths)
        else:
            return len(self.indices)

    @abstractmethod
    def _load_example_(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        r"""This method defines how inputs, labels, and meta info are loaded from disk.

        Args:
            idx (int): index of data point.

        Returns:
            inputs (Dict[str, torch.Tensor]): the inputs to the model.
            labels (Dict[str, torch.Tensor]): the ground truth for the current inputs.
            meta_info (Dict[str, Any]): the meta info for the current data point.
        """
        raise NotImplementedError("[ERROR] _load_example_ not implemented for abstract base class.")

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        idx = self.indices[idx] if self.indices is not None else idx
        inputs, labels, meta_info = self._load_example_(idx)
        example = {
            'inputs': inputs,
            'labels': labels,
            'meta_info': meta_info,
        }
        example = self.transforms(example)
        return example
