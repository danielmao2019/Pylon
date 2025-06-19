from typing import List, Dict, Optional, Any
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from utils.monitor.gpu_status import GPUStatus, get_server_gpus_info
from utils.automation.ssh_utils import _ssh_pool


class GPUMonitor:

    def __init__(self, gpu_indices_by_server: Optional[Dict[str, List[int]]] = None, timeout: int = 5):
        """
        Initialize GPU monitor with GPU indices organized by server.

        Args:
            gpu_indices_by_server: Dictionary mapping server names to lists of GPU indices, or None for localhost
            timeout: SSH command timeout in seconds
        """
        self.gpus_by_server = self._init_gpu_status(gpu_indices_by_server)
        self.timeout = timeout
        self.servers = list(self.gpus_by_server.keys())
        self.monitor_thread: Optional[threading.Thread] = None

        # Do one update first
        self.ssh_pool = _ssh_pool
        self._update()

    def _init_gpu_status(self, gpu_indices_by_server: Optional[Dict[str, List[int]]]) -> Dict[str, List[GPUStatus]]:
        """Initialize GPUStatus objects from GPU indices organized by server.

        Args:
            gpu_indices_by_server: Dictionary mapping server names to lists of GPU indices, or None for localhost

        Returns:
            Dictionary mapping server names to lists of GPUStatus objects
        """
        gpus_by_server = {}
        
        if gpu_indices_by_server is None:
            # Handle localhost case - get physical GPU index
            if torch.cuda.is_available():
                device_index = torch.cuda.current_device()
                cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
                if cuda_visible_devices:
                    visible_devices = [int(d.strip()) for d in cuda_visible_devices.split(',')]
                    physical_device_index = visible_devices[device_index]
                else:
                    physical_device_index = device_index
                
                gpu_status: GPUStatus = {
                    'server': 'localhost',
                    'index': physical_device_index,
                    'max_memory': None,
                    'processes': None,
                    'window_size': 10,  # Default window size
                    'memory_window': None,
                    'util_window': None,
                    'memory_stats': None,
                    'util_stats': None,
                    'connected': False,
                }
                gpus_by_server['localhost'] = [gpu_status]
        else:
            # Regular dictionary of server -> indices
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
                self._update()

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _update(self):
        """Updates information for all GPUs using batched queries per server"""
        with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
            list(executor.map(self._update_single_server, self.servers))

    def _update_single_server(self, server: str) -> None:
        """Update all GPUs on a single server using batched queries"""
        indices = [gpu['index'] for gpu in self.gpus_by_server[server]]

        # Get batched GPU info for all GPUs on this server
        server_gpus_info = get_server_gpus_info(server, indices, self.ssh_pool, timeout=self.timeout)

        # Update each GPU with the batched results
        for idx in indices:
            self._update_single_gpu(self.gpus_by_server[server][idx], server_gpus_info[idx])

    def _update_single_gpu(self, gpu_status: GPUStatus, gpu_info: Dict[str, Any]) -> None:
        """Update a single GPU with the provided info"""
        # Update connection status
        gpu_status['connected'] = gpu_info['connected']

        # If query failed, set all fields to None except server and index
        if not gpu_info['connected']:
            gpu_status['max_memory'] = None
            gpu_status['processes'] = None
            gpu_status['memory_window'] = None
            gpu_status['util_window'] = None
            gpu_status['memory_stats'] = None
            gpu_status['util_stats'] = None
            return

        # Update processes
        gpu_status['processes'] = gpu_info['processes']

        # Update max memory
        gpu_status['max_memory'] = gpu_info['max_memory']

        # Initialize windows if they are None (GPU just reconnected)
        if gpu_status['memory_window'] is None:
            gpu_status['memory_window'] = []
        if gpu_status['util_window'] is None:
            gpu_status['util_window'] = []

        # Update rolling windows
        gpu_status['memory_window'].append(gpu_info['current_memory'])
        gpu_status['util_window'].append(gpu_info['current_util'])

        # Trim windows if needed
        if len(gpu_status['memory_window']) > gpu_status['window_size']:
            gpu_status['memory_window'] = gpu_status['memory_window'][-gpu_status['window_size']:]
        if len(gpu_status['util_window']) > gpu_status['window_size']:
            gpu_status['util_window'] = gpu_status['util_window'][-gpu_status['window_size']:]

        # Update stats
        gpu_status['memory_stats'] = {
            'min': min(gpu_status['memory_window']),
            'max': max(gpu_status['memory_window']),
            'avg': sum(gpu_status['memory_window']) / len(gpu_status['memory_window']),
        }
        gpu_status['util_stats'] = {
            'min': min(gpu_status['util_window']),
            'max': max(gpu_status['util_window']),
            'avg': sum(gpu_status['util_window']) / len(gpu_status['util_window']),
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
            for server_gpus in self.gpus_by_server.values()
            for gpu in server_gpus
        }

    @property
    def gpus(self) -> List[GPUStatus]:
        """Get all GPUs"""
        return [
            gpu for server_gpus in self.gpus_by_server.values() for gpu in server_gpus
        ]

    @property
    def connected_gpus(self) -> List[GPUStatus]:
        """Get all connected GPUs"""
        return [
            gpu for server_gpus in self.gpus_by_server.values() for gpu in server_gpus
            if gpu['connected']
        ]

    @property
    def disconnected_gpus(self) -> Dict[str, List[int]]:
        """Get all disconnected GPUs"""
        return {
            server: [gpu['index'] for gpu in server_gpus if not gpu['connected']]
            for server, server_gpus in self.gpus_by_server.items()
        }

    def get_all_running_commands(self) -> List[str]:
        """Get all running commands on all servers"""
        all_gpus = [
            gpu for server_gpus in self.gpus_by_server.values() for gpu in server_gpus
            if gpu['connected']
        ]
        all_processes = [
            process for gpu in all_gpus for process in gpu['processes']
            if process['cmd'].startswith('python main.py --config-filepath')
        ]
        return all_processes

    def log_stats(self, logger):
        """Logs status of all monitored GPUs"""
        stats = self._check()
        assert len(stats) == 1, "Only support single GPU training for now."
        stats = list(stats.values())[0]
        logger.update_buffer(stats)
