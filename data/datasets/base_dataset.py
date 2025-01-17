from typing import Tuple, List, Dict, Union, Any, Optional
from abc import ABC, abstractmethod
import itertools
import copy
import subprocess
import os
import random
import torch
from data.transforms.compose import Compose
from utils.input_checks import check_read_dir
from utils.builder import build_from_config
from utils.ops import apply_tensor_op


class BaseDataset(torch.utils.data.Dataset, ABC):

    SPLIT_OPTIONS: List[str]
    DATASET_SIZE: Dict[str, int]
    INPUT_NAMES: List[str]
    LABEL_NAMES: List[str]
    SHA1SUM: str

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: Optional[Union[str, Tuple[float, ...]]] = None,
        indices: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        transforms_cfg: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        device: Optional[torch.device] = torch.device('cuda'),
        check_sha1sum: Optional[bool] = False,
    ) -> None:
        r"""
        Args:
            use_cache (bool): controls whether loaded data points stays in RAM. Default: True.
        """
        torch.multiprocessing.set_start_method('spawn', force=True)
        # input checks
        if data_root is not None:
            self.data_root = check_read_dir(path=data_root)
        # sanity check
        self.check_sha1sum = check_sha1sum
        self._sanity_check()
        # initialize
        super(BaseDataset, self).__init__()
        if use_cache:
            self.cache: List[Dict[str, Dict[str, Any]]] = []
        else:
            self.cache = None
        self.device = device
        self._init_transforms_(transforms_cfg=transforms_cfg)
        self._init_annotations_all_(split=split, indices=indices)

    def _sanity_check(self) -> None:
        assert self.SPLIT_OPTIONS is not None
        if hasattr(self, 'DATASET_SIZE') and self.DATASET_SIZE is not None:
            assert set(self.SPLIT_OPTIONS) == set(self.DATASET_SIZE.keys())
        assert self.INPUT_NAMES is not None
        assert self.LABEL_NAMES is not None
        assert set(self.INPUT_NAMES) & set(self.LABEL_NAMES) == set(), \
            f"{self.INPUT_NAMES=}, {self.LABEL_NAMES=}, {set(self.INPUT_NAMES) & set(self.LABEL_NAMES)=}"
        if self.check_sha1sum and os.path.islink(self.data_root) and hasattr(self, 'SHA1SUM') and self.SHA1SUM is not None:
            cmd = ' '.join([
                'find', os.readlink(self.data_root), '-type', 'f', '-print0', '|',
                'sort', '-z', '|', 'xargs', '-0', 'sha1sum', '|', 'sha1sum',
            ])
            result = subprocess.getoutput(cmd)
            sha1sum = result.split(' ')[0]
            assert sha1sum == self.SHA1SUM, f"{sha1sum=}, {self.SHA1SUM=}"

    def _init_transforms_(self, transforms_cfg: Optional[Dict[str, Any]]) -> None:
        if transforms_cfg is None:
            transforms_cfg = {
                'class': Compose,
                'args': {
                    'transforms': [],
                },
            }
        self.transforms = build_from_config(transforms_cfg)

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
                if hasattr(self, 'DATASET_SIZE') and self.DATASET_SIZE is not None:
                    assert len(self) == self.DATASET_SIZE[split], f"{len(self)=}, {self.DATASET_SIZE[split]=}, {split=}"
                # take subset by indices
                if indices is not None:
                    assert type(self.annotations) == list, f"{type(self.annotations)=}"
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
                    if hasattr(self, 'DATASET_SIZE') and self.DATASET_SIZE is not None:
                        assert len(self) == self.DATASET_SIZE[option], f"{len(self)=}, {self.DATASET_SIZE[option]=}, {option=}"
                    # take subset by indices
                    if indices is not None and option in indices:
                        assert type(self.annotations) == list, f"{type(self.annotations)=}"
                        self.annotations = [self.annotations[idx] for idx in indices[option]]
                    # prepare to split
                    split_subset = copy.deepcopy(self)
                    del split_subset.split_subsets
                    self.split_subsets[option] = split_subset
            else:
                # input checks
                assert type(split) == tuple, f"{type(split)=}"
                assert len(split) == len(self.SPLIT_OPTIONS), f"{len(split)=}, {len(self.SPLIT_OPTIONS)=}"
                assert all(type(elem) == float for elem in split)
                assert abs(sum(split) - 1.0) < 0.01, f"{sum(split)=}"
                # initialize full list of annotations
                self._init_annotations_(split=None)
                # take subset by indices
                if indices is not None:
                    assert type(self.annotations) == list, f"{type(self.annotations)=}"
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
    def _load_datapoint(self, idx: int) -> Tuple[
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
        raise NotImplementedError("[ERROR] _load_datapoint not implemented for abstract base class.")

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        if self.cache is None or idx >= len(self.cache) or self.cache[idx] is None:
            inputs, labels, meta_info = self._load_datapoint(idx)
            datapoint = {
                'inputs': inputs,
                'labels': labels,
                'meta_info': meta_info,
            }
            datapoint = self.transforms(datapoint)
            if self.cache is not None:
                if idx >= len(self.cache):
                    self.cache += [None] * (idx - len(self.cache) + 1)
                self.cache[idx] = copy.deepcopy(datapoint)
        else:
            datapoint = copy.deepcopy(self.cache[idx])
        datapoint = apply_tensor_op(func=lambda x: x.to(self.device), inputs=datapoint)
        return datapoint
