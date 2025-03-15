from typing import Tuple, List, Dict, Union, Any, Optional
from abc import ABC, abstractmethod
import itertools
import copy
import subprocess
import os
import random
import torch
import threading
import logging
import time
from collections import OrderedDict
import json
import hashlib
from pathlib import Path
import psutil
from data.transforms.compose import Compose
from utils.input_checks import check_read_dir
from utils.builders import build_from_config
from utils.ops import apply_tensor_op


class DatasetCache:
    """A thread-safe cache manager for dataset items with LRU eviction policy."""
    
    def __init__(
        self,
        max_memory_percent: float = 80.0,
        persist_path: Optional[str] = None,
        version: str = "1.0",
        enable_validation: bool = True,
    ):
        """
        Args:
            max_memory_percent (float): Maximum percentage of system memory to use
            persist_path (str, optional): Path to persist cache to disk
            version (str): Cache version for compatibility checking
            enable_validation (bool): Whether to enable checksum validation
        """
        self.max_memory_percent = max_memory_percent
        self.persist_path = Path(persist_path) if persist_path else None
        self.version = version
        self.enable_validation = enable_validation
        
        # Initialize cache structures
        self.cache = OrderedDict()
        self.checksums = {}  # Store checksums for validation
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.validation_failures = 0
        
        # Load persisted cache if available
        if self.persist_path and self.persist_path.exists():
            self._load_persistent_cache()
            
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _compute_checksum(self, value: Dict[str, Any]) -> str:
        """Compute a checksum for a cached item."""
        # Convert tensors to numpy for consistent hashing
        def prepare_for_hash(item):
            if isinstance(item, torch.Tensor):
                return item.cpu().numpy().tobytes()
            elif isinstance(item, dict):
                return {k: prepare_for_hash(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [prepare_for_hash(x) for x in item]
            return item

        # Prepare data for hashing
        hashable_data = prepare_for_hash(value)
        return hashlib.sha256(str(hashable_data).encode()).hexdigest()

    def _validate_item(self, key: int, value: Dict[str, Any]) -> bool:
        """Validate a cached item against its stored checksum."""
        if not self.enable_validation:
            return True
            
        if key not in self.checksums:
            return False
            
        current_checksum = self._compute_checksum(value)
        is_valid = current_checksum == self.checksums[key]
        
        if not is_valid:
            self.validation_failures += 1
            self.logger.warning(f"Cache validation failed for key {key}")
            
        return is_valid

    def get(self, key: int) -> Optional[Dict[str, Any]]:
        """Thread-safe cache retrieval with LRU update and validation."""
        with self.lock:
            if key in self.cache:
                value = self.cache[key]
                
                # Validate item before returning
                if self._validate_item(key, value):
                    self.hits += 1
                    # Update LRU order
                    self.cache.pop(key)
                    self.cache[key] = value
                    return copy.deepcopy(value)
                else:
                    # Remove invalid item
                    self.cache.pop(key)
                    self.checksums.pop(key, None)
                    
            self.misses += 1
            return None
            
    def put(self, key: int, value: Dict[str, Any]) -> None:
        """Thread-safe cache insertion with memory management and checksum computation."""
        with self.lock:
            # Check memory usage and evict if needed
            while self._get_memory_usage() > self.max_memory_percent and self.cache:
                evicted_key, _ = self.cache.popitem(last=False)  # Remove oldest item
                self.checksums.pop(evicted_key, None)
                
            # Store item and its checksum
            value_copy = copy.deepcopy(value)
            self.cache[key] = value_copy
            if self.enable_validation:
                self.checksums[key] = self._compute_checksum(value_copy)
            
            # Periodically persist cache
            if self.persist_path and len(self.cache) % 1000 == 0:
                self._persist_cache()
                
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        return psutil.Process().memory_percent()
        
    def _persist_cache(self) -> None:
        """Persist cache and checksums to disk."""
        if not self.persist_path:
            return
            
        cache_data = {
            "version": self.version,
            "timestamp": time.time(),
            "hits": self.hits,
            "misses": self.misses,
            "validation_failures": self.validation_failures,
            "cache": self.cache,
            "checksums": self.checksums
        }
        
        temp_path = self.persist_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'wb') as f:
                torch.save(cache_data, f)
            temp_path.rename(self.persist_path)
        except Exception as e:
            self.logger.error(f"Failed to persist cache: {e}")
            if temp_path.exists():
                temp_path.unlink()
                
    def _load_persistent_cache(self) -> None:
        """Load cache and checksums from disk with validation."""
        try:
            cache_data = torch.load(self.persist_path)
            if cache_data["version"] != self.version:
                self.logger.warning("Cache version mismatch, starting fresh")
                return
                
            # Load basic stats
            self.hits = cache_data["hits"]
            self.misses = cache_data["misses"]
            self.validation_failures = cache_data.get("validation_failures", 0)
            
            # Load and validate cache items
            self.cache = OrderedDict()
            self.checksums = {}
            
            for key, value in cache_data["cache"].items():
                stored_checksum = cache_data["checksums"].get(key)
                if not self.enable_validation or stored_checksum is None:
                    self.cache[key] = value
                    self.checksums[key] = self._compute_checksum(value)
                else:
                    current_checksum = self._compute_checksum(value)
                    if current_checksum == stored_checksum:
                        self.cache[key] = value
                        self.checksums[key] = stored_checksum
                    else:
                        self.validation_failures += 1
                        self.logger.warning(f"Validation failed for cached item {key} during load")
            
            self.logger.info(
                f"Loaded cache with {len(self.cache)} items. "
                f"Hit rate: {self.hits/(self.hits+self.misses+1e-6):.2%}, "
                f"Validation failures: {self.validation_failures}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load persistent cache: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including validation metrics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / (self.hits + self.misses + 1e-6),
                "memory_usage": self._get_memory_usage(),
                "validation_failures": self.validation_failures
            }


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
        cache_dir: Optional[str] = None,
        max_cache_memory_percent: float = 80.0,
        device: Optional[Union[torch.device, str]] = torch.device('cuda'),
        check_sha1sum: Optional[bool] = False,
    ) -> None:
        """
        Args:
            use_cache (bool): controls whether loaded data points stays in RAM. Default: True
            cache_dir (str, optional): directory to persist cache to disk
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
            persist_path = None
            if cache_dir:
                # Create unique cache file name based on dataset parameters
                cache_hash = hashlib.sha256(
                    f"{self.data_root}_{split}_{indices}_{transforms_cfg}".encode()
                ).hexdigest()[:8]
                persist_path = os.path.join(cache_dir, f"dataset_cache_{cache_hash}.pt")
                
            self.cache = DatasetCache(
                max_memory_percent=max_cache_memory_percent,
                persist_path=persist_path,
                version="1.0"
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
        assert split is None or type(split) in [str, tuple], f"{type(split)=}"
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
        if self.cache is not None:
            cached_item = self.cache.get(idx)
            if cached_item is not None:
                return apply_tensor_op(
                    func=lambda x: x.to(self.device), 
                    inputs=cached_item
                )
        
        # Load and process item
        inputs, labels, meta_info = self._load_datapoint(idx)
        datapoint = {
            'inputs': inputs,
            'labels': labels,
            'meta_info': meta_info,
        }
        datapoint = self.transforms(datapoint)
        
        # Cache the processed item
        if self.cache is not None:
            self.cache.put(idx, copy.deepcopy(datapoint))
            
        return apply_tensor_op(
            func=lambda x: x.to(self.device), 
            inputs=datapoint
        )

    def visualize(self, output_dir: str) -> None:
        raise NotImplementedError("Class method 'visualize' not implemented.")

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if caching is enabled."""
        if self.cache is not None:
            return self.cache.get_stats()
        return None
