import os
import queue
import threading
from typing import Any, Dict, Optional

from data.cache.base_cache import BaseCache
from data.cache.cpu_dataset_cache import CPUDatasetCache
from data.cache.disk_dataset_cache import DiskDatasetCache


class CombinedDatasetCache(BaseCache):
    """Unified cache interface combining CPU memory and disk caching with hierarchy: CPU -> Disk -> Source."""

    def __init__(
        self,
        data_root: str,
        version_hash: str,
        use_cpu_cache: bool = True,
        use_disk_cache: bool = True,
        max_cpu_memory_percent: float = 80.0,
        enable_cpu_validation: bool = False,
        enable_disk_validation: bool = False,
        dataset_class_name: Optional[str] = None,
        version_dict: Optional[Dict[str, Any]] = None,
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
            dataset_class_name: Name of the dataset class for metadata
            version_dict: Dictionary containing version parameters for metadata
        """
        super().__init__()
        data_root = os.path.realpath(data_root)
        assert os.path.isabs(
            data_root
        ), f"data_root must resolve to absolute path, got {data_root=}"
        self.use_cpu_cache = use_cpu_cache
        self.use_disk_cache = use_disk_cache
        self.cache_dir = f"{data_root}_cache"
        assert os.path.isabs(
            self.cache_dir
        ), f"cache_dir must be absolute, got {self.cache_dir=}"
        self.version_hash = version_hash
        self.version_dir = os.path.join(self.cache_dir, version_hash)
        assert os.path.isabs(
            self.version_dir
        ), f"version_dir must be absolute, got {self.version_dir=}"
        os.makedirs(self.version_dir, exist_ok=True)

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
            self.disk_cache = DiskDatasetCache(
                cache_dir=self.cache_dir,
                version_hash=version_hash,
                enable_validation=enable_disk_validation,
                dataset_class_name=dataset_class_name,
                version_dict=version_dict,
            )
        else:
            self.disk_cache = None
        self._start_put_worker()

    def _start_put_worker(self) -> None:
        self._worker_exception: Optional[BaseException] = None
        self._put_queue: queue.Queue = queue.Queue()
        self._put_thread = threading.Thread(
            target=self._put_worker,
            daemon=True,
            name="combined_cache_put_worker",
        )
        self._put_thread.start()

    def _check_worker_health(self) -> None:
        assert (
            self._worker_exception is None
        ), f"Cache put worker failed with {self._worker_exception}"
        assert (
            self._put_thread.is_alive()
        ), "Cache put worker thread stopped unexpectedly"

    def _wait_for_pending_puts(self) -> None:
        self._check_worker_health()
        self._put_queue.join()
        self._check_worker_health()

    def _put_worker(self) -> None:
        while True:
            task = self._put_queue.get()
            if self._worker_exception is not None:
                self._put_queue.task_done()
                continue

            try:
                assert isinstance(task, dict)
                assert 'cache_filepath' in task and 'value' in task
                cache_filepath = task['cache_filepath']
                value = task['value']
                assert isinstance(cache_filepath, str)

                if self.cpu_cache is not None:
                    self.cpu_cache.put(
                        value=value,
                        cache_filepath=cache_filepath,
                    )

                if self.disk_cache is not None:
                    self.disk_cache.put(
                        value=value,
                        cache_filepath=cache_filepath,
                    )
            except BaseException as exc:
                self._worker_exception = exc
            finally:
                self._put_queue.task_done()

    def _retrieve_from_caches(
        self,
        cache_filepath: str,
        device: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if self.cpu_cache is not None:
            value = self.cpu_cache.get(
                cache_filepath=cache_filepath,
            )
            if value is not None:
                return value

        if self.disk_cache is not None:
            value = self.disk_cache.get(
                device=device,
                cache_filepath=cache_filepath,
            )
            if value is not None and self.cpu_cache is not None:
                self.cpu_cache.put(
                    value=value,
                    cache_filepath=cache_filepath,
                )
            return value

        return None

    def get(
        self,
        cache_filepath: str,
        device: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get item from cache hierarchy: CPU -> Disk -> None.

        If found in disk but not CPU, populate CPU cache.
        """
        assert isinstance(
            cache_filepath, str
        ), f"cache_filepath must be str, got {type(cache_filepath)=}"
        self._check_worker_health()

        cached_value = self._retrieve_from_caches(
            cache_filepath=cache_filepath,
            device=device,
        )
        if cached_value is not None:
            return cached_value

        self._wait_for_pending_puts()
        return self._retrieve_from_caches(
            cache_filepath=cache_filepath,
            device=device,
        )

    def put(
        self,
        value: Dict[str, Any],
        cache_filepath: str,
    ) -> None:
        """
        Store item in both CPU and disk caches.

        Args:
            value: Raw datapoint with 'inputs', 'labels', 'meta_info' keys
        """
        assert isinstance(
            cache_filepath, str
        ), f"cache_filepath must be str, got {type(cache_filepath)=}"
        if self.cpu_cache is None and self.disk_cache is None:
            return
        self._check_worker_health()
        self._put_queue.put(
            {
                'value': value,
                'cache_filepath': cache_filepath,
            }
        )

    def clear(self) -> None:
        """Clear both CPU and disk caches (BaseCache API)."""
        self._wait_for_pending_puts()
        if self.cpu_cache is not None:
            self.cpu_cache = CPUDatasetCache(
                max_memory_percent=self.cpu_cache.max_memory_percent,
                enable_validation=self.cpu_cache.enable_validation,
            )
        if self.disk_cache is not None:
            self.disk_cache.clear()

    def get_size(self) -> int:
        """Get total number of cached items (disk preferred)."""
        self._wait_for_pending_puts()
        if self.disk_cache is not None:
            return self.disk_cache.get_size()
        if self.cpu_cache is not None:
            return len(self.cpu_cache.cache)
        return 0

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        if '_put_queue' in state:
            state['_put_queue'] = None
        if '_put_thread' in state:
            state['_put_thread'] = None
        state['_worker_exception'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        self._start_put_worker()
