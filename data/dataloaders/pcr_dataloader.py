import os
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
                if key in self.cache:
                    return self.cache.get()
                else:
                    actual_datapoints = [original_dataset[idx] for idx in datapoints]
                    batched_datapoints = collator(actual_datapoints)
                    self.cache.put(key, batched_datapoints)
            super().__init__(self, dataset=index_dataset, collate_fn=new_collator, **kwargs)
        else:
            super().__init__(self, dataset=dataset, collate_fn=collator, **kwargs)

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
