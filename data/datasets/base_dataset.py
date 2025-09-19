from typing import Tuple, List, Dict, Union, Any, Optional
from abc import ABC, abstractmethod
import itertools
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
        split: Optional[str] = None,
        split_percentages: Optional[Tuple[float, ...]] = None,
        indices: Optional[List[int]] = None,
        transforms_cfg: Optional[Dict[str, Any]] = None,
        base_seed: int = 0,
        use_cpu_cache: bool = True,
        use_disk_cache: bool = True,
        max_cache_memory_percent: float = 80.0,
        enable_cpu_validation: bool = False,
        enable_disk_validation: bool = False,
        overwrite_cache: bool = False,
        device: Optional[Union[torch.device, str]] = torch.device('cuda'),
        check_sha1sum: Optional[bool] = False,
    ) -> None:
        """
        Args:
            data_root (str, optional): path to the dataset root directory
            split (str, optional): which split to initialize ('train', 'val', 'test', etc.). None means load everything.
            split_percentages (tuple, optional): percentages for random split when predefined splits don't exist
            indices (list, optional): subset of indices to use from the split
            transforms_cfg (dict, optional): configuration for data transforms
            base_seed (int): seed for deterministic behavior. Default: 0
            use_cpu_cache (bool): controls whether loaded data points stays in RAM. Default: True
            use_disk_cache (bool): controls whether to use disk cache. Default: True
            max_cache_memory_percent (float): maximum percentage of system memory to use for cache
            enable_cpu_validation (bool): validate cached data consistency. Default: False
            enable_disk_validation (bool): validate disk cached data consistency. Default: False
            overwrite_cache (bool): force regeneration of cached datapoints. Default: False
            device (torch.device): target device for tensors. Default: cuda
            check_sha1sum (bool): verify dataset integrity with SHA1 checksum. Default: False
        """
        torch.multiprocessing.set_start_method('spawn', force=True)

        # input checks
        if data_root is not None:
            self.data_root = check_read_dir(path=data_root)

        # Validate overwrite_cache
        assert isinstance(overwrite_cache, bool), f"overwrite_cache must be bool, got {type(overwrite_cache)=}"
        if overwrite_cache:
            assert use_cpu_cache or use_disk_cache, "When overwrite_cache=True, at least one of use_cpu_cache or use_disk_cache must be True"
        self.overwrite_cache = overwrite_cache

        # initialize
        super(BaseDataset, self).__init__()
        self._init_split(split=split, split_percentages=split_percentages)
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

    def _init_split(self, split: Optional[str], split_percentages: Optional[Tuple[float, ...]]) -> None:
        if split is None:
            # split=None means load everything (no predefined splits, no percentage splits)
            assert split_percentages is None, "Cannot use split_percentages when split=None"
            assert not (hasattr(self, 'DATASET_SIZE') and isinstance(self.DATASET_SIZE, dict)), \
                "Cannot use dict DATASET_SIZE when split=None (implies predefined splits exist)"
            assert not (hasattr(self, 'CLASS_DIST') and isinstance(self.CLASS_DIST, dict)), \
                "Cannot use dict CLASS_DIST when split=None (implies predefined splits exist)"
            self.split = None
            return

        assert isinstance(split, str), f"split must be string or None, got {type(split)=}"
        assert split in self.SPLIT_OPTIONS, f"{split=} not in {self.SPLIT_OPTIONS=}"

        self.split = split

        if split_percentages is not None:
            assert isinstance(split_percentages, tuple), f"split_percentages must be tuple, got {type(split_percentages)=}"
            assert len(split_percentages) == len(self.SPLIT_OPTIONS), f"{split_percentages=}, {self.SPLIT_OPTIONS=}"
            assert all(isinstance(x, (int, float)) for x in split_percentages), f"All percentages must be numeric, got {split_percentages=}"
            assert abs(sum(split_percentages) - 1.0) < 0.01, f"Percentages must sum to 1.0, got sum={sum(split_percentages)}"
            self.split_percentages = split_percentages

        # Normalize DATASET_SIZE to always be int for the selected split
        if hasattr(self, 'DATASET_SIZE') and self.DATASET_SIZE is not None:
            if isinstance(self.DATASET_SIZE, dict):
                assert set(self.DATASET_SIZE.keys()).issubset(set(self.SPLIT_OPTIONS)), \
                    f"DATASET_SIZE keys {self.DATASET_SIZE.keys()} != SPLIT_OPTIONS {self.SPLIT_OPTIONS}"
                assert split_percentages is None, "Cannot use split_percentages with dict DATASET_SIZE"
                self.DATASET_SIZE = self.DATASET_SIZE[self.split]
            elif isinstance(self.DATASET_SIZE, int):
                assert split_percentages is not None, "int DATASET_SIZE requires split_percentages"
            else:
                assert False, f"DATASET_SIZE must be dict or int, got {type(self.DATASET_SIZE)=}"

        if hasattr(self, 'CLASS_DIST') and isinstance(self.CLASS_DIST, dict):
            assert split_percentages is None, "Cannot use split_percentages with CLASS_DIST"
            assert set(self.CLASS_DIST.keys()) == set(self.SPLIT_OPTIONS), f"CLASS_DIST keys {self.CLASS_DIST.keys()} != SPLIT_OPTIONS {self.SPLIT_OPTIONS}"
            assert all(isinstance(x, list) for x in self.CLASS_DIST.values()), f"All CLASS_DIST values must be lists"
            self.CLASS_DIST = self.CLASS_DIST[self.split]

    def _init_indices(self, indices: Optional[List[int]]) -> None:
        assert indices is None or isinstance(indices, list), f"indices must be None or list, got {type(indices)=}"
        if isinstance(indices, list):
            assert all(isinstance(x, int) for x in indices), f"All indices must be integers, got {indices=}"
        self.indices = indices

    def _init_annotations_all_splits(self) -> None:
        """Initialize annotations for the single specified split or everything if split=None.

        If split_percentages is provided, performs deterministic random split first,
        then initializes only the requested split's annotations.
        """
        # Initialize annotations and check dataset size (always needed)
        self._init_annotations()
        self._check_dataset_size()

        if self.split is None:
            # Load everything - no split processing needed
            pass
        elif hasattr(self, 'split_percentages') and self.split_percentages is not None:
            # Perform deterministic random split
            assert isinstance(self.split, str), "split must be string when using split_percentages"
            assert self.split in self.SPLIT_OPTIONS, f"{self.split=} not in {self.SPLIT_OPTIONS=}"

            sizes = tuple(int(percent * len(self.annotations)) for percent in self.split_percentages)
            cutoffs = [0] + list(itertools.accumulate(sizes))

            # Use deterministic shuffle with base_seed for reproducibility
            rng = random.Random(self.base_seed)
            rng.shuffle(self.annotations)

            # Extract only the requested split's annotations
            split_idx = self.SPLIT_OPTIONS.index(self.split)
            self.annotations = self.annotations[cutoffs[split_idx]:cutoffs[split_idx+1]]
        else:
            # Predefined split - annotations already loaded correctly
            assert isinstance(self.split, str), "split must be string when using predefined splits"
            assert self.split in self.SPLIT_OPTIONS, f"{self.split=} not in {self.SPLIT_OPTIONS=}"

        # Apply indices filtering if provided
        self._filter_annotations_by_indices()

    @abstractmethod
    def _init_annotations(self) -> None:
        raise NotImplementedError("_init_annotations not implemented for abstract base class.")

    def _check_dataset_size(self) -> None:
        if not hasattr(self, 'DATASET_SIZE') or self.DATASET_SIZE is None:
            return

        assert isinstance(self.DATASET_SIZE, int), f"DATASET_SIZE should be normalized to int in _init_split, got {type(self.DATASET_SIZE)=}"
        assert len(self.annotations) == self.DATASET_SIZE, f"{len(self.annotations)=}, {self.DATASET_SIZE=}"

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

            # For datasets without data_root (e.g., random datasets), use a default location
            # For datasets with soft links, resolve to real path to ensure cache is in target location (e.g., /pub not /home)
            if hasattr(self, 'data_root'):
                data_root_for_cache = self.data_root
                if os.path.islink(data_root_for_cache):
                    data_root_for_cache = os.path.realpath(data_root_for_cache)
            else:
                # Use dataset class name for default location when no data_root is provided
                data_root_for_cache = f'/tmp/cache/{self.__class__.__name__.lower()}'

            self.cache = CombinedDatasetCache(
                data_root=data_root_for_cache,
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

        if hasattr(self, 'split_percentages') and self.split_percentages is not None:
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
        from utils.determinism.hash_utils import convert_to_seed
        self.base_seed = convert_to_seed(seed)

    def _load_datapoint_with_cache(self, idx: int) -> Dict[str, Any]:
        """Load datapoint with caching support.

        Args:
            idx: Index of the datapoint to load

        Returns:
            Dictionary containing inputs, labels, and meta_info
        """
        # Try to get raw datapoint from cache first (unless overwrite_cache is True)
        raw_datapoint = None
        if self.cache is not None and not self.overwrite_cache:
            raw_datapoint = self.cache.get(idx, device=self.device)

        # If not in cache, or cache is disabled, or overwrite_cache is True, load from disk and cache it
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

        return raw_datapoint

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        # Load datapoint with caching
        raw_datapoint = self._load_datapoint_with_cache(idx)

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
