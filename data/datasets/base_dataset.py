from typing import Tuple, List, Dict, Union, Any, Optional
from abc import ABC, abstractmethod
import copy
import itertools
import random
import torch
from ..transforms.compose import Compose
from utils.input_checks import check_read_dir
from utils.builder import build_from_config


class BaseDataset(ABC, torch.utils.data.Dataset):

    SPLIT_OPTIONS: List[str] = None
    INPUT_NAMES: List[str] = None
    LABEL_NAMES: List[str] = None

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: Optional[Union[str, Tuple[float, ...]]] = None,
        indices: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        transforms: Optional[list] = None,
    ) -> None:
        super(BaseDataset, self).__init__()
        # sanity checks
        assert self.SPLIT_OPTIONS is not None
        assert self.INPUT_NAMES is not None
        assert self.LABEL_NAMES is not None
        assert set(self.INPUT_NAMES) & set(self.LABEL_NAMES) == set(), f"{set(self.INPUT_NAMES) & set(self.LABEL_NAMES)=}"
        # initialize
        if data_root is not None:
            self.data_root = check_read_dir(path=data_root)
        self._init_transform_(transforms=transforms)
        self._init_annotations_all_(split=split, indices=indices)

    def _init_transform_(self, transforms: Optional[list]) -> None:
        if transforms is None:
            transforms = {
                'class': Compose,
                'args': {
                    'transforms': [],
                },
            }
        self.transforms = build_from_config(transforms)

    def _init_annotations_all_(
        self,
        split: Optional[Union[str, Tuple[float, ...]]],
        indices: Optional[Union[List[int], Dict[str, List[int]]]],
    ) -> None:
        r"""
        Args:
            split (str or Tuple[float, ...] or None): if str, then initialize only the specified split.
                if Tuple[float, ...], then randomly split whole dataset according to specified percentages.
                if None, then initialize all splits.
            indices (List[int] or Dict[str, List[int]] or None): a list of indices defining the subset
                of interest.
        """
        # initialize annotations
        if type(split) == str:
            if split in self.SPLIT_OPTIONS:
                # initialize full list of annotations
                self._init_annotations_(split=split)
                assert type(self.annotations) == list, f"{type(self.annotations)=}"
                # take subset by indices
                if indices is not None:
                    assert type(indices) == list, f"{type(indices)=}"
                    assert all(type(elem) == int for elem in indices)
                    self.annotations = [self.annotations[idx] for idx in indices]
            else:
                self.annotations = []
        else:
            self.split_subsets: Dict[str, BaseDataset] = {}
            if split is None:
                # input checks
                if indices is not None:
                    assert type(indices) == dict, f"{type(indices)=}"
                    assert all(type(val) == list for val in indices.values())
                    assert all(all(type(elem) == int for elem in val) for val in indices.values())
                # construct split subsets
                for option in self.SPLIT_OPTIONS:
                    # initialize full list of annotations
                    self._init_annotations_(split=option)
                    assert type(self.annotations) == list, f"{type(self.annotations)=}"
                    # take subset by indices
                    if indices is not None and option in indices:
                        self.annotations = [self.annotations[idx] for idx in indices[option]]
                    # prepare to split
                    self.split_subsets[option] = copy.deepcopy(self)
            else:
                # input checks
                assert type(split) == tuple, f"{type(split)=}"
                assert len(split) == len(self.SPLIT_OPTIONS), f"{len(split)=}, {len(self.SPLIT_OPTIONS)=}"
                assert all(type(elem) == float for elem in split)
                assert abs(sum(split) - 1.0) < 0.01, f"{sum(split)=}"
                # initialize full list of annotations
                self._init_annotations_(split=None)
                assert type(self.annotations) == list, f"{type(self.annotations)=}"
                # take subset by indices
                if indices is not None:
                    assert type(indices) == list, f"{type(indices)=}"
                    assert all(type(elem) == int for elem in indices)
                    self.annotations = [self.annotations[idx] for idx in indices]
                # prepare to split
                sizes = tuple(int(percent * len(self.annotations)) for percent in split)
                cutoffs = [0] + list(itertools.accumulate(sizes))
                random.shuffle(self.annotations)
                for idx, option in enumerate(self.SPLIT_OPTIONS):
                    split_subset = copy.deepcopy(self)
                    split_subset.annotations = self.annotations[cutoffs[idx]:cutoffs[idx+1]]
                    del split_subset.split_subsets
                    self.split_subsets[option] = split_subset

    @abstractmethod
    def _init_annotations_(self, split: Optional[str]) -> None:
        raise NotImplementedError("_init_annotations_ not implemented for abstract base class.")

    def __len__(self):
        return len(self.annotations)

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
        inputs, labels, meta_info = self._load_example_(idx)
        example = {
            'inputs': inputs,
            'labels': labels,
            'meta_info': meta_info,
        }
        example = self.transforms(example)
        return example
