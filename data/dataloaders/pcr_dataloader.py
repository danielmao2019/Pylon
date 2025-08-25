import os
import json
import xxhash
from typing import List, Dict, Any
from data.cache.combined_dataset_cache import CombinedDatasetCache
from data.dataloaders.base_dataloader import BaseDataLoader


class PCRCachedCollator:
    """Picklable collator wrapper for PCR dataloader with caching functionality."""
    
    def __init__(self, original_dataset, collator, cache):
        self.original_dataset = original_dataset
        self.collator = collator
        self.cache = cache
    
    def __call__(self, datapoints: List[int]):
        assert isinstance(datapoints, list)
        assert len(datapoints) == 1
        assert isinstance(datapoints[0], int)
        key = datapoints[0]
        assert self.cache is not None
        cached_result = self.cache.get(key)
        if cached_result is not None:
            return cached_result
        else:
            actual_datapoints = [self.original_dataset[idx] for idx in datapoints]
            batched_datapoints = self.collator(actual_datapoints)
            self.cache.put(key, batched_datapoints)
            return batched_datapoints


class PCRDataloader(BaseDataLoader):

    def __init__(
        self,
        dataset,
        collator,
        use_cpu_cache,
        use_disk_cache,
        max_cache_memory_percent,
        enable_cpu_validation,
        enable_disk_validation,
        **kwargs,
    ) -> None:
        self._init_cache(
            dataset=dataset,
            collator=collator,
            use_cpu_cache=use_cpu_cache,
            use_disk_cache=use_disk_cache,
            max_cache_memory_percent=max_cache_memory_percent,
            enable_cpu_validation=enable_cpu_validation,
            enable_disk_validation=enable_disk_validation,
        )
        if self.cache is not None:
            index_dataset = list(range(len(dataset)))
            cached_collator = PCRCachedCollator(dataset, collator, self.cache)
            super().__init__(dataset=index_dataset, collate_fn=cached_collator, **kwargs)
        else:
            super().__init__(dataset=dataset, collate_fn=collator, **kwargs)

    def _init_cache(
        self,
        dataset,
        collator,
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
            version_hash = self.get_cache_version_hash(dataset, collator)
            
            # For datasets without data_root (e.g., random datasets), use a default location
            # For datasets with soft links, resolve to real path to ensure cache is in target location (e.g., /pub not /home)
            if hasattr(dataset, 'data_root'):
                data_root_for_cache = dataset.data_root
                if os.path.islink(data_root_for_cache):
                    data_root_for_cache = os.path.realpath(data_root_for_cache)
            else:
                # Use dataset class name for default location when no data_root is provided
                data_root_for_cache = f'/tmp/cache/{dataset.__class__.__name__.lower()}'
            
            self.cache = CombinedDatasetCache(
                data_root=data_root_for_cache,
                version_hash=version_hash,
                use_cpu_cache=use_cpu_cache,
                use_disk_cache=use_disk_cache,
                max_cpu_memory_percent=max_cache_memory_percent,
                enable_cpu_validation=enable_cpu_validation,
                enable_disk_validation=enable_disk_validation,
                dataset_class_name=dataset.__class__.__name__,
                version_dict=self._get_cache_version_dict(dataset, collator),
            )
        else:
            self.cache = None
    
    def _get_cache_version_dict(self, dataset, collator) -> Dict[str, Any]:
        """Return parameters that affect dataloader cache content for cache versioning.
        
        Base implementation provides common fields. Subclasses should call super()
        and add their specific parameters.
        
        Args:
            dataset: The dataset being used
            collator: The collator being used
            
        Returns:
            Dict containing version parameters for this PCR dataloader configuration
        """
        return {
            'dataloader_class': self.__class__.__name__,
            'dataset_version': dataset.get_cache_version_hash(),
        }
    
    def get_cache_version_hash(self, dataset, collator):
        """Generate deterministic hash from dataloader configuration."""
        version_dict = self._get_cache_version_dict(dataset, collator)
        hash_str = json.dumps(version_dict, sort_keys=True)
        return xxhash.xxh64(hash_str.encode()).hexdigest()[:16]
