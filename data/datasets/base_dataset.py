from typing import Tuple, List, Dict, Union, Any, Optional
from abc import ABC, abstractmethod
import itertools
import copy
import subprocess
import os
import random
import torch
from data.cache import DatasetCache
from data.transforms.compose import Compose
from utils.input_checks import check_read_dir
from utils.builders import build_from_config
from utils.ops import apply_tensor_op


class BaseDataset(torch.utils.data.Dataset, ABC):

    SPLIT_OPTIONS: List[str]
    DATASET_SIZE: Union[Dict[str, int], int]
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
        max_cache_memory_percent: float = 80.0,
        device: Optional[Union[torch.device, str]] = torch.device('cuda'),
        check_sha1sum: Optional[bool] = False,
    ) -> None:
        """
        Args:
            use_cache (bool): controls whether loaded data points stays in RAM. Default: True
            max_cache_memory_percent (float): maximum percentage of system memory to use for cache
        """
        torch.multiprocessing.set_start_method('spawn', force=True)

        # input checks
        if data_root is not None:
            self.data_root = check_read_dir(path=data_root)

        # initialize
        super(BaseDataset, self).__init__()
        self._init_split(split=split)
        self._init_indices(indices=indices)
        self._init_transforms(transforms_cfg=transforms_cfg)

        # Initialize cache
        if use_cache:
            self.cache = DatasetCache(
                max_memory_percent=max_cache_memory_percent,
                enable_validation=True
            )
        else:
            self.cache = None

        self._init_device(device)

        # sanity check
        self.check_sha1sum = check_sha1sum
        self._sanity_check()

        # initialize annotations at the end because of splits
        self._init_annotations_all_splits()

    def _init_split(self, split: Optional[Union[str, Tuple[float, ...]]]) -> None:
        assert split is None or isinstance(split, (str, tuple)), f"{type(split)=}"
        if type(split) == tuple:
            assert len(split) == len(self.SPLIT_OPTIONS), f"{split=}, {self.SPLIT_OPTIONS=}"
            assert all(type(x) in [int, float] for x in split)
            assert abs(sum(split) - 1.0) < 0.01, f"{sum(self.split)=}"
            self.split_percentages = split
        else:
            self.split = split
        if hasattr(self, 'CLASS_DIST') and type(self.CLASS_DIST) == dict:
            assert set(self.CLASS_DIST.keys()) == set(self.SPLIT_OPTIONS)
            assert all(type(x) == list for x in self.CLASS_DIST.values())
            self.CLASS_DIST = self.CLASS_DIST[self.split]

    def _init_indices(self, indices: Optional[Union[List[int], Dict[str, List[int]]]]) -> None:
        # type check
        if hasattr(self, 'split') and type(self.split) == str:
            assert indices is None or type(indices) == list
            if type(indices) == list:
                assert all(type(x) == int for x in indices)
            self.indices = indices
        elif hasattr(self, 'split') and self.split is None:
            assert indices is None or type(indices) == dict
            if type(indices) == dict:
                assert set(indices.keys()).issubset(set(self.SPLIT_OPTIONS)), \
                    f"{indices.keys()=}, {self.SPLIT_OPTIONS=}"
                for value in indices.values():
                    assert type(value) == list
                    assert all(type(x) == int for x in value)
            self.split_indices = indices
        else:
            assert not hasattr(self, 'split') and hasattr(self, 'split_percentages')

    def _init_annotations_all_splits(self) -> None:
        r"""
        Args:
            split (str or Tuple[float, ...] or None): if str, then initialize only the specified split.
                if Tuple[float, ...], then randomly split whole dataset according to specified percentages.
                if None, then initialize all splits.
            indices (List[int] or Dict[str, List[int]] or None): a list of indices defining the subset
                of interest.
        """
        # initialize annotations
        if hasattr(self, 'split') and type(self.split) == str:
            if self.split in self.SPLIT_OPTIONS:
                # initialize annotations
                self._init_annotations()
                # sanity check
                if hasattr(self, 'DATASET_SIZE') and self.DATASET_SIZE is not None:
                    assert type(self.DATASET_SIZE) == dict, f"{type(self.DATASET_SIZE)=}"
                    assert len(self) == self.DATASET_SIZE[self.split], f"{len(self)=}, {self.DATASET_SIZE[self.split]=}, {self.split=}"
                # take subset by indices
                self._filter_annotations_by_indices()
            else:
                self.annotations = []
        else:
            self.split_subsets: Dict[str, BaseDataset] = {}
            if hasattr(self, 'split') and self.split is None:
                assert not hasattr(self, 'indices') and hasattr(self, 'split_indices')
                assert self.split_indices is None or type(self.split_indices) == dict
                # construct split subsets
                for split in self.SPLIT_OPTIONS:
                    # initialize split and indices
                    self.split = split
                    self.indices = self.split_indices.get(split, None) if self.split_indices else None
                    # initialize annotations
                    self._init_annotations()
                    self._check_dataset_size()
                    self._filter_annotations_by_indices()
                    # prepare to split
                    split_subset = copy.deepcopy(self)
                    del split_subset.split_indices
                    del split_subset.split_subsets
                    self.split_subsets[split] = split_subset
                self.split = None
                del self.indices
            else:
                assert not hasattr(self, 'split') and hasattr(self, 'split_percentages')
                assert not hasattr(self, 'indices') and not hasattr(self, 'split_indices')
                # initialize annotations
                self._init_annotations()
                self._check_dataset_size()
                # prepare to split
                sizes = tuple(int(percent * len(self.annotations)) for percent in self.split_percentages)
                cutoffs = [0] + list(itertools.accumulate(sizes))
                random.shuffle(self.annotations)
                for idx, split in enumerate(self.SPLIT_OPTIONS):
                    self.split = split
                    split_subset = copy.deepcopy(self)
                    split_subset.annotations = self.annotations[cutoffs[idx]:cutoffs[idx+1]]
                    del split_subset.split_percentages
                    del split_subset.split_subsets
                    self.split_subsets[split] = split_subset
                del self.split

    @abstractmethod
    def _init_annotations(self) -> None:
        raise NotImplementedError("_init_annotations not implemented for abstract base class.")

    def _check_dataset_size(self) -> None:
        if not hasattr(self, 'DATASET_SIZE') or self.DATASET_SIZE is None:
            return
        assert type(self.DATASET_SIZE) in [dict, int]
        if type(self.DATASET_SIZE) == int:
            assert type(self.split) == tuple
        if type(self.DATASET_SIZE) == dict:
            assert set(self.DATASET_SIZE.keys()).issubset(set(self.SPLIT_OPTIONS)), \
                f"{self.DATASET_SIZE.keys()=}, {self.SPLIT_OPTIONS=}"
            assert type(self.split) == str
            self.DATASET_SIZE = self.DATASET_SIZE[self.split]
        assert len(self) == len(self.annotations) == self.DATASET_SIZE, \
            f"{len(self)=}, {len(self.annotations)=}, {self.DATASET_SIZE=}"

    def _filter_annotations_by_indices(self) -> None:
        if self.indices is None:
            return
        assert type(self.annotations) == list, f"{type(self.annotations)=}"
        self.annotations = [self.annotations[idx % len(self.annotations)] for idx in self.indices]

    def _init_transforms(self, transforms_cfg: Optional[Dict[str, Any]]) -> None:
        if transforms_cfg is None:
            transforms_cfg = {
                'class': Compose,
                'args': {
                    'transforms': [],
                },
            }
        self.transforms = build_from_config(transforms_cfg)

    def _init_device(self, device: Union[str, torch.device]) -> None:
        if type(device) == str:
            device = torch.device(device)
        assert type(device) == torch.device, f"{type(device)=}"
        self.device = device

    def _sanity_check(self) -> None:
        assert self.SPLIT_OPTIONS is not None
        if hasattr(self, 'DATASET_SIZE') and self.DATASET_SIZE is not None:
            assert set(self.SPLIT_OPTIONS) == set(self.DATASET_SIZE.keys())
        assert self.INPUT_NAMES is not None
        assert self.LABEL_NAMES is not None
        assert set(self.INPUT_NAMES) & set(self.LABEL_NAMES) == set(), \
            f"{self.INPUT_NAMES=}, {self.LABEL_NAMES=}, {set(self.INPUT_NAMES) & set(self.LABEL_NAMES)=}"
        if self.check_sha1sum and hasattr(self, 'SHA1SUM') and self.SHA1SUM is not None:
            cmd = ' '.join([
                'find', os.readlink(self.data_root) if os.path.islink(self.data_root) else self.data_root,
                '-type', 'f', '-execdir', 'sha1sum', '{}', '+', '|',
                'sort', '|', 'sha1sum',
            ])
            result = subprocess.getoutput(cmd)
            sha1sum = result.split(' ')[0]
            assert sha1sum == self.SHA1SUM, f"{sha1sum=}, {self.SHA1SUM=}"

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
        # Try to get raw datapoint from cache first
        raw_datapoint = None
        if self.cache is not None:
            raw_datapoint = self.cache.get(idx)

        # If not in cache, load from disk and cache it
        if raw_datapoint is None:
            # Load raw datapoint
            inputs, labels, meta_info = self._load_datapoint(idx)
            raw_datapoint = {
                'inputs': inputs,
                'labels': labels,
                'meta_info': meta_info,
            }
            # Cache the raw datapoint
            if self.cache is not None:
                self.cache.put(idx, raw_datapoint)

        # Apply transforms to the raw datapoint (whether from cache or freshly loaded)
        transformed_datapoint = self.transforms(raw_datapoint)

        # Move to device
        return apply_tensor_op(
            func=lambda x: x.to(self.device),
            inputs=transformed_datapoint
        )

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if caching is enabled."""
        if self.cache is not None:
            return self.cache.get_stats()
        return None
