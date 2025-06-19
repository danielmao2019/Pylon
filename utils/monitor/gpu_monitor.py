from typing import List, Dict, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
from utils.monitor.gpu_status import GPUStatus, get_gpu_info
from utils.automation.ssh_utils import SSHConnectionPool


class GPUMonitor:

    def __init__(self, gpu_indices_by_server: Dict[str, List[int]], ssh_pool: SSHConnectionPool, timeout: int = 5):
        """
        Initialize GPU monitor with GPU indices organized by server.

        Args:
            gpu_indices_by_server: Dictionary mapping server names to lists of GPU indices
            ssh_pool: SSH connection pool instance (required)
            timeout: SSH command timeout in seconds
        """
        self.ssh_pool = ssh_pool
        self.timeout = timeout
        self.monitor_thread: Optional[threading.Thread] = None

        # Initialize GPUStatus objects from indices
        self.gpus_by_server = self._init_gpu_status(gpu_indices_by_server)

        # Do one update first
        self._update_gpu_info_batched()

    def _init_gpu_status(self, gpu_indices_by_server: Dict[str, List[int]]) -> Dict[str, List[GPUStatus]]:
        """Initialize GPUStatus objects from GPU indices organized by server.

        Args:
            gpu_indices_by_server: Dictionary mapping server names to lists of GPU indices

        Returns:
            Dictionary mapping server names to lists of GPUStatus objects
        """
        gpus_by_server = {}

        for server, indices in gpu_indices_by_server.items():
            server_gpus = []
            for gpu_idx in indices:
                gpu_status: GPUStatus = {
                    'server': server,
                    'index': gpu_idx,
                    'max_memory': None,
                    'processes': None,
                    'window_size': 10,  # Default window size
                    'memory_window': None,
                    'util_window': None,
                    'memory_stats': None,
                    'util_stats': None,
                    'connected': False,
                }
                server_gpus.append(gpu_status)
            gpus_by_server[server] = server_gpus

        return gpus_by_server

    def start(self):
        """Starts background monitoring thread that continuously updates GPU info"""
        def monitor_loop():
            while True:
                self._update_gpu_info_batched()

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _update_gpu_info_batched(self):
        """Updates information for all GPUs using batched queries per server"""
        # Use ThreadPoolExecutor to query multiple servers in parallel
        with ThreadPoolExecutor(max_workers=len(self.gpus_by_server)) as executor:
            # Submit batched queries for each server
            future_to_server = {
                executor.submit(self._update_server_gpus, server, gpus): server
                for server, gpus in self.gpus_by_server.items()
            }

            # Collect results
            for future in future_to_server:
                try:
                    future.result(timeout=self.timeout * 2)  # Allow extra time for batched queries
                except Exception as e:
                    server = future_to_server[future]
                    print(f"ERROR: Failed to update GPUs for server {server}: {e}")

    def _update_server_gpus(self, server: str, server_gpus: List[GPUStatus]):
        """Update all GPUs on a single server using batched queries"""
        gpu_indices = [gpu['index'] for gpu in server_gpus]

        # Get batched GPU info for all GPUs on this server
        batch_results = get_gpu_info(server, gpu_indices, self.ssh_pool, timeout=self.timeout)

        # Update each GPU with the batched results
        for gpu in server_gpus:
            gpu_idx = gpu['index']
            current_info = batch_results[gpu_idx]
            self._update_single_gpu(gpu, current_info)

    def _update_single_gpu(self, gpu: GPUStatus, current_info: Dict):
        """Update a single GPU with the provided info"""
        # Update connection status
        gpu['connected'] = current_info['connected']

        # If query failed, set all fields to None except server and index
        if not current_info['connected']:
            gpu['max_memory'] = None
            gpu['processes'] = None
            gpu['memory_window'] = None
            gpu['util_window'] = None
            gpu['memory_stats'] = None
            gpu['util_stats'] = None
            return

        # Update processes
        gpu['processes'] = current_info['processes']

        # Update max memory
        gpu['max_memory'] = current_info['max_memory']

        # Initialize windows if they are None (GPU just reconnected)
        if gpu['memory_window'] is None:
            gpu['memory_window'] = []
        if gpu['util_window'] is None:
            gpu['util_window'] = []

        # Update rolling windows
        gpu['memory_window'].append(current_info['current_memory'])
        gpu['util_window'].append(current_info['current_util'])

        # Trim windows if needed
        if len(gpu['memory_window']) > gpu['window_size']:
            gpu['memory_window'] = gpu['memory_window'][-gpu['window_size']:]
        if len(gpu['util_window']) > gpu['window_size']:
            gpu['util_window'] = gpu['util_window'][-gpu['window_size']:]

        # Update stats
        gpu['memory_stats'] = {
            'min': min(gpu['memory_window']),
            'max': max(gpu['memory_window']),
            'avg': sum(gpu['memory_window']) / len(gpu['memory_window']),
        }
        gpu['util_stats'] = {
            'min': min(gpu['util_window']),
            'max': max(gpu['util_window']),
            'avg': sum(gpu['util_window']) / len(gpu['util_window']),
        }

    def _check(self) -> Dict:
        """Returns current status of all GPUs without rolling windows"""
        all_gpus = []
        for server_gpus in self.gpus_by_server.values():
            all_gpus.extend(server_gpus)

        return {
            f"{gpu['server']}:{gpu['index']}": {
                'server': gpu['server'],
                'index': gpu['index'],
                'max_memory': gpu['max_memory'],
                'memory_min': gpu['memory_stats']['min'] if gpu['memory_stats'] is not None else None,
                'memory_max': gpu['memory_stats']['max'] if gpu['memory_stats'] is not None else None,
                'memory_avg': gpu['memory_stats']['avg'] if gpu['memory_stats'] is not None else None,
                'util_min': gpu['util_stats']['min'] if gpu['util_stats'] is not None else None,
                'util_max': gpu['util_stats']['max'] if gpu['util_stats'] is not None else None,
                'util_avg': gpu['util_stats']['avg'] if gpu['util_stats'] is not None else None,
                'connected': gpu['connected'],
            }
            for gpu in all_gpus
        }

    def log_stats(self, logger):
        """Logs status of all monitored GPUs"""
        stats = self._check()
        assert len(stats) == 1, "Only support single GPU training for now."
        stats = list(stats.values())[0]
        logger.update_buffer(stats)
