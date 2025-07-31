"""
DATA.CACHE API
"""

from data.cache.base_cache import BaseCache
from data.cache.cpu_dataset_cache import CPUDatasetCache
from data.cache.disk_dataset_cache import DiskDatasetCache
from data.cache.combined_dataset_cache import CombinedDatasetCache


__all__ = [
    'BaseCache',
    'CPUDatasetCache',
    'DiskDatasetCache',
    'CombinedDatasetCache',
]
