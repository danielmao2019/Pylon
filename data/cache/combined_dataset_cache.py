from typing import Dict, Any, Optional
from data.cache.cpu_dataset_cache import CPUDatasetCache
from data.cache.disk_dataset_cache import DiskDatasetCache


class CombinedDatasetCache:
    """Unified cache interface combining CPU memory and disk caching with hierarchy: CPU → Disk → Source."""

    def __init__(
        self,
        data_root: str,
        version_hash: str,
        use_cpu_cache: bool = True,
        use_disk_cache: bool = True,
        max_cpu_memory_percent: float = 80.0,
        enable_cpu_validation: bool = False,
        enable_disk_validation: bool = False,
    ):
        """
        Args:
            data_root: Path to dataset root directory
            version_hash: Hash identifying this dataset version
            use_cpu_cache: Whether to enable CPU memory caching
            use_disk_cache: Whether to enable disk caching
            max_cpu_memory_percent: Maximum percentage of system memory for CPU cache
            enable_cpu_validation: Whether to enable checksum validation for CPU cache
            enable_disk_validation: Whether to enable checksum validation for disk cache
        """
        self.use_cpu_cache = use_cpu_cache
        self.use_disk_cache = use_disk_cache
        
        # Initialize CPU cache
        if use_cpu_cache:
            self.cpu_cache = CPUDatasetCache(
                max_memory_percent=max_cpu_memory_percent,
                enable_validation=enable_cpu_validation,
            )
        else:
            self.cpu_cache = None
        
        # Initialize disk cache
        if use_disk_cache:
            cache_dir = f"{data_root}_cache"
            self.disk_cache = DiskDatasetCache(
                cache_dir=cache_dir,
                version_hash=version_hash,
                enable_validation=enable_disk_validation,
            )
        else:
            self.disk_cache = None

    def get(self, idx: int, device: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get item from cache hierarchy: CPU → Disk → None.
        
        If found in disk but not CPU, populate CPU cache.
        """
        # Try CPU cache first (fastest)
        if self.cpu_cache is not None:
            value = self.cpu_cache.get(idx, device=device)
            if value is not None:
                return value
        
        # Try disk cache
        if self.disk_cache is not None:
            value = self.disk_cache.get(idx, device=device)
            if value is not None:
                # Populate CPU cache for future access (store on CPU for cache)
                if self.cpu_cache is not None:
                    # Store in CPU cache without device transfer (will be applied on next CPU cache hit)
                    cpu_value = self.disk_cache.get(idx, device='cpu') if device != 'cpu' else value
                    self.cpu_cache.put(idx, cpu_value)
                return value
        
        # Not found in either cache
        return None

    def put(self, idx: int, value: Dict[str, Any]) -> None:
        """
        Store item in both CPU and disk caches.
        
        Args:
            idx: Index of the datapoint
            value: Raw datapoint with 'inputs', 'labels', 'meta_info' keys
        """
        # Store in CPU cache
        if self.cpu_cache is not None:
            self.cpu_cache.put(idx, value)
        
        # Store in disk cache
        if self.disk_cache is not None:
            self.disk_cache.put(idx, value)

    def exists_on_disk(self, idx: int) -> bool:
        """Check if item exists in disk cache."""
        if self.disk_cache is not None:
            return self.disk_cache.exists(idx)
        return False

    def clear_cpu_cache(self) -> None:
        """Clear CPU cache only."""
        if self.cpu_cache is not None:
            # Clear by recreating the cache
            self.cpu_cache = CPUDatasetCache(
                max_memory_percent=self.cpu_cache.max_memory_percent,
                enable_validation=self.cpu_cache.enable_validation,
            )

    def clear_disk_cache(self) -> None:
        """Clear disk cache only."""
        if self.disk_cache is not None:
            self.disk_cache.clear()

    def clear_all(self) -> None:
        """Clear both CPU and disk caches."""
        self.clear_cpu_cache()
        self.clear_disk_cache()

    def get_cpu_size(self) -> int:
        """Get number of items in CPU cache."""
        if self.cpu_cache is not None:
            return len(self.cpu_cache.cache)
        return 0

    def get_disk_size(self) -> int:
        """Get number of items in disk cache."""
        if self.disk_cache is not None:
            return self.disk_cache.get_size()
        return 0

    def get_info(self) -> Dict[str, Any]:
        """Get cache information."""
        info = {
            'use_cpu_cache': self.use_cpu_cache,
            'use_disk_cache': self.use_disk_cache,
            'cpu_cache_size': self.get_cpu_size(),
            'disk_cache_size': self.get_disk_size(),
        }
        
        if self.disk_cache is not None:
            info['cache_dir'] = self.disk_cache.cache_dir
            info['version_hash'] = self.disk_cache.version_hash
        
        return info
