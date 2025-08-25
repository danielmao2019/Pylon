import os
from typing import List
from data.cache.combined_dataset_cache import CombinedDatasetCache
from data.dataloaders.base_dataloader import BaseDataLoader


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
            original_dataset = dataset
            index_dataset = list(range(len(dataset)))
            def new_collator(datapoints: List[int]):
                assert isinstance(datapoints, list)
                assert len(datapoints) == 1
                assert isinstance(datapoints[0], int)
                key = datapoints[0]
                assert self.cache is not None
                cached_result = self.cache.get(key)
                if cached_result is not None:
                    return cached_result
                else:
                    actual_datapoints = [original_dataset[idx] for idx in datapoints]
                    batched_datapoints = collator(actual_datapoints)
                    self.cache.put(key, batched_datapoints)
                    return batched_datapoints
            super().__init__(dataset=index_dataset, collate_fn=new_collator, **kwargs)
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
            version_hash = self._get_cache_version_hash(dataset, collator)
            
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
    
    def _get_cache_version_hash(self, dataset, collator):
        """Generate version hash for this dataset/collator configuration."""
        dataset_version = getattr(dataset, 'get_cache_version', lambda: 'v1')()
        collator_version = getattr(collator, 'get_cache_version', lambda: 'v1')()
        return f"{dataset_version}_{collator_version}"
    
    def _get_cache_version_dict(self, dataset, collator):
        """Generate version dictionary for cache validation."""
        return {
            'dataset_class': dataset.__class__.__name__,
            'dataset_version': getattr(dataset, 'get_cache_version', lambda: 'v1')(),
            'collator_class': collator.__class__.__name__,
            'collator_version': getattr(collator, 'get_cache_version', lambda: 'v1')(),
        }
