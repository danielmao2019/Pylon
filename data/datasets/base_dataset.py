from typing import Tuple, List, Dict, Union, Any, Optional
from abc import ABC, abstractmethod
import itertools
import copy
import subprocess
import os
import random
import json
import xxhash
import torch
from data.cache import CombinedDatasetCache
from data.transforms.compose import Compose
from utils.input_checks import check_read_dir
from utils.builders import build_from_config
from utils.ops import apply_tensor_op


class BaseDataset(torch.utils.data.Dataset, ABC):

    SPLIT_OPTIONS: List[str]
    DATASET_SIZE: Optional[Union[Dict[str, int], int]]
    INPUT_NAMES: List[str]
    LABEL_NAMES: List[str]
    SHA1SUM: Optional[str]

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: Optional[Union[str, Tuple[float, ...]]] = None,
        indices: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        transforms_cfg: Optional[Dict[str, Any]] = None,
        base_seed: int = 0,
        use_cpu_cache: bool = True,
        use_disk_cache: bool = True,
        max_cache_memory_percent: float = 80.0,
        enable_cpu_validation: bool = False,
        enable_disk_validation: bool = False,
        device: Optional[Union[torch.device, str]] = torch.device('cuda'),
        check_sha1sum: Optional[bool] = False,
    ) -> None:
        """
        Args:
            use_cpu_cache (bool): controls whether loaded data points stays in RAM. Default: True
            use_disk_cache (bool): controls whether to use disk cache. Default: True
            max_cache_memory_percent (float): maximum percentage of system memory to use for cache
            base_seed (int): seed for deterministic behavior. Default: 0
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
        self.set_base_seed(base_seed)
        self._init_cache(
            use_cpu_cache=use_cpu_cache,
            use_disk_cache=use_disk_cache,
            max_cache_memory_percent=max_cache_memory_percent,
            enable_cpu_validation=enable_cpu_validation,
            enable_disk_validation=enable_disk_validation,
        )
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
            assert abs(sum(split) - 1.0) < 0.01, f"{sum(split)=}"
            self.split_percentages = split
        else:
            self.split = split
        if hasattr(self, 'CLASS_DIST') and isinstance(self.CLASS_DIST, dict):
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
                if hasattr(self, 'DATASET_SIZE') and self.DATASET_SIZE is not None and isinstance(self.DATASET_SIZE, dict):
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
            assert hasattr(self, 'split_percentages'), "int DATASET_SIZE requires split_percentages"
            assert not hasattr(self, 'split'), "int DATASET_SIZE should not have split attribute"
        if type(self.DATASET_SIZE) == dict:
            assert set(self.DATASET_SIZE.keys()).issubset(set(self.SPLIT_OPTIONS)), \
                f"{self.DATASET_SIZE.keys()=}, {self.SPLIT_OPTIONS=}"
            assert hasattr(self, 'split') and isinstance(self.split, str), "dict DATASET_SIZE requires string split"
            assert not hasattr(self, 'split_percentages'), "dict DATASET_SIZE should not have split_percentages attribute"
            self.DATASET_SIZE = self.DATASET_SIZE[self.split]
        assert len(self) == len(self.annotations) == self.DATASET_SIZE, \
            f"{len(self)=}, {len(self.annotations)=}, {self.DATASET_SIZE=}"

    def _filter_annotations_by_indices(self) -> None:
        if self.indices is None:
            return
        assert type(self.annotations) == list, f"{type(self.annotations)=}"
        self.annotations = [self.annotations[idx % len(self.annotations)] for idx in self.indices]

    def _init_transforms(self, transforms_cfg: Optional[Dict[str, Any]]) -> None:
        if transforms_cfg is None or transforms_cfg == {}:
            transforms_cfg = {
                'class': Compose,
                'args': {
                    'transforms': [],
                },
            }
        self.transforms = build_from_config(transforms_cfg)

    def _init_cache(
        self,
        use_cpu_cache: bool,
        use_disk_cache: bool,
        max_cache_memory_percent: float,
        enable_cpu_validation: bool,
        enable_disk_validation: bool,
    ) -> None:
        assert isinstance(use_cpu_cache, bool), f"{type(use_cpu_cache)=}"
        assert isinstance(use_disk_cache, bool), f"{type(use_disk_cache)=}"
        assert isinstance(max_cache_memory_percent, float), f"{type(max_cache_memory_percent)=}"
        assert 0.0 <= max_cache_memory_percent <= 100.0, f"{max_cache_memory_percent=}"
        
        if use_cpu_cache or use_disk_cache:
            # Generate version hash for this dataset configuration
            version_hash = self.get_cache_version_hash()
            
            # For datasets without data_root (e.g., random datasets), use a default cache location
            # For datasets with soft links, resolve to real path to ensure cache is in target location (e.g., /pub not /home)
            if hasattr(self, 'data_root'):
                cache_data_root = self.data_root
                if os.path.islink(cache_data_root):
                    cache_data_root = os.path.realpath(cache_data_root)
            else:
                # Use dataset class name for cache directory when no data_root is provided
                cache_data_root = f'/tmp/cache/{self.__class__.__name__.lower()}'
            
            self.cache = CombinedDatasetCache(
                data_root=cache_data_root,
                version_hash=version_hash,
                use_cpu_cache=use_cpu_cache,
                use_disk_cache=use_disk_cache,
                max_cpu_memory_percent=max_cache_memory_percent,
                enable_cpu_validation=enable_cpu_validation,
                enable_disk_validation=enable_disk_validation,
                dataset_class_name=self.__class__.__name__,
                version_dict=self._get_cache_version_dict(),
            )
        else:
            self.cache = None

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning.
        
        Subclasses should override this method to add their specific parameters.
        
        Note: data_root is intentionally excluded from the version dict to ensure
        cache hashes are stable across different filesystem locations (e.g., soft links).
        """
        version_dict = {
            'class_name': self.__class__.__name__,
        }
        
        # NOTE: We explicitly DO NOT include data_root in the version dict
        # This ensures that the same dataset accessed through different paths
        # (e.g., soft links, relocated datasets) will have the same cache hash
        
        # Add split information
        if hasattr(self, 'split') and self.split is not None:
            version_dict['split'] = self.split
        elif hasattr(self, 'split_percentages'):
            version_dict['split_percentages'] = self.split_percentages
        
        # Add base parameters that affect dataset content
        if hasattr(self, 'base_seed') and self.base_seed is not None:
            version_dict['base_seed'] = self.base_seed
        
        if hasattr(self, 'indices') and self.indices is not None:
            version_dict['indices'] = self.indices
        
        return version_dict
    
    def get_cache_version_hash(self) -> str:
        """Generate deterministic hash from dataset configuration."""
        version_dict = self._get_cache_version_dict()
        hash_str = json.dumps(version_dict, sort_keys=True)
        return xxhash.xxh64(hash_str.encode()).hexdigest()[:16]

    def _init_device(self, device: Union[str, torch.device]) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device), f"{type(device)=}"
        self.device = device

    def _sanity_check(self) -> None:
        assert hasattr(self, 'SPLIT_OPTIONS') and self.SPLIT_OPTIONS is not None
        if hasattr(self, 'DATASET_SIZE') and self.DATASET_SIZE is not None and isinstance(self.DATASET_SIZE, dict):
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

    def set_base_seed(self, seed: Any) -> None:
        """Set the base seed for the dataset."""
        if not isinstance(seed, int):
            seed = hash(seed) % (2**32)  # Ensure it's a 32-bit integer
        self.base_seed = seed

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        # Try to get raw datapoint from cache first
        raw_datapoint = None
        if self.cache is not None:
            raw_datapoint = self.cache.get(idx, device=self.device)

        # If not in cache, load from disk and cache it
        if raw_datapoint is None:
            # Load raw datapoint
            inputs, labels, meta_info = self._load_datapoint(idx)

            # Ensure 'idx' hasn't been added by concrete class, then add it
            assert 'idx' not in meta_info, f"Dataset class should not manually add 'idx' to meta_info. Found 'idx' in meta_info: {meta_info.keys()}"
            meta_info['idx'] = idx

            raw_datapoint = {
                'inputs': inputs,
                'labels': labels,
                'meta_info': meta_info,
            }
            # Cache the raw datapoint
            if self.cache is not None:
                self.cache.put(idx, raw_datapoint)

        # Apply device transfer only if not already on target device (cache may have done this)
        datapoint = apply_tensor_op(func=lambda x: x.to(self.device), inputs=raw_datapoint)
        transformed_datapoint = self.transforms(datapoint, seed=(self.base_seed, idx))
        return transformed_datapoint

    @staticmethod
    @abstractmethod
    def display_datapoint(
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> Optional['html.Div']:
        """Create custom display for this dataset's datapoints.
        
        This method allows datasets to provide custom visualization logic
        that will be used by the data viewer instead of the default display functions.
        
        Args:
            datapoint: Dictionary containing inputs, labels, and meta_info from dataset
            class_labels: Optional dictionary mapping class indices to label names
            camera_state: Optional dictionary containing camera position state for 3D visualizations
            settings_3d: Optional dictionary containing 3D visualization settings
            
        Returns:
            Optional html.Div: Custom HTML layout for displaying this datapoint.
            Return None to indicate fallback to predefined display functions.
            
        Note:
            This is an abstract static method that must be implemented by all dataset subclasses.
            Return None if you want to use the default display functions based on dataset type.
        """
        raise NotImplementedError("display_datapoint not implemented for abstract base class.")
