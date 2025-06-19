from typing import List, Dict, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
from utils.monitor.gpu_status import GPUStatus, get_gpu_info, get_all_gpu_info_batched, get_ssh_pool_status
from utils.automation.ssh_utils import SSHConnectionPool


class GPUMonitor:

    def __init__(self, gpus_by_server: Dict[str, List[GPUStatus]], timeout: int = 5, ssh_pool: Optional[SSHConnectionPool] = None):
        """
        Initialize GPU monitor with GPUs organized by server.
        
        Args:
            gpus_by_server: Dictionary mapping server names to lists of GPUStatus objects
            timeout: SSH command timeout in seconds
            ssh_pool: Optional SSH connection pool instance
        """
        self.gpus_by_server = gpus_by_server
        self.monitor_thread: Optional[threading.Thread] = None
        self.timeout = timeout
        
        # Use provided SSH pool or create a new one
        self.ssh_pool = ssh_pool if ssh_pool is not None else SSHConnectionPool()

        # Do one update first
        self._update_gpu_info_batched()

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
        
        try:
            # Get batched GPU info for all GPUs on this server
            batch_results = get_all_gpu_info_batched(server, gpu_indices, self.ssh_pool, timeout=self.timeout)
            
            # Update each GPU with the batched results
            for gpu in server_gpus:
                gpu_idx = gpu['index']
                if gpu_idx in batch_results:
                    current_info = batch_results[gpu_idx]
                    self._update_single_gpu(gpu, current_info)
                else:
                    # Fallback to individual query if not in batch results
                    self._update_gpu_info(gpu)
                    
        except Exception as e:
            print(f"ERROR: Failed to update server {server} GPUs {gpu_indices}: {e}")
            # Fallback to individual queries
            for gpu in server_gpus:
                self._update_gpu_info(gpu)

    def _update_single_gpu(self, gpu: GPUStatus, current_info: Dict):
        """Update a single GPU with the provided info"""
        # Update connection status
        gpu['connected'] = current_info['success']

        # If query failed, set all fields to None except server and index
        if not current_info['success']:
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
            for gpu in self.gpus
        }

    def log_stats(self, logger):
        """Logs status of all monitored GPUs"""
        stats = self._check()
        assert len(stats) == 1, "Only support single GPU training for now."
        stats = list(stats.values())[0]
        logger.update_buffer(stats)

    def get_ssh_pool_stats(self) -> Dict[str, Dict[str, int]]:
        """Get SSH connection pool statistics for monitoring"""
        return get_ssh_pool_status(self.ssh_pool)

    def get_connection_efficiency_stats(self) -> Dict[str, float]:
        """Get connection efficiency statistics"""
        pool_stats = self.get_ssh_pool_stats()
        efficiency_stats = {}
        
        for server, stats in pool_stats.items():
            if stats['max_connections'] > 0:
                utilization = stats['active_connections'] / stats['max_connections']
                pool_usage = stats['pool_size'] / stats['max_connections']
                efficiency_stats[server] = {
                    'connection_utilization': utilization,
                    'pool_usage': pool_usage,
                    'active_connections': stats['active_connections'],
                    'pool_size': stats['pool_size'],
                    'max_connections': stats['max_connections']
                }
        
        return efficiency_stats
