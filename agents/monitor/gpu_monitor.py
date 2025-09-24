from typing import List, Dict, Optional, Any
import os
import torch
from agents.monitor.base_monitor import BaseMonitor
from agents.monitor.gpu_status import GPUStatus
from agents.monitor.process_info import ProcessInfo, get_all_processes
from agents.connector.pool import SSHConnectionPool
from utils.timeout import with_timeout


class GPUMonitor(BaseMonitor[GPUStatus]):

    def __init__(self, gpu_indices_by_server: Optional[Dict[str, List[int]]] = None, timeout: int = 5):
        """
        Initialize GPU monitor with GPU indices organized by server.

        Args:
            gpu_indices_by_server: Dictionary mapping server names to lists of GPU indices, or None for localhost
            timeout: SSH command timeout in seconds
        """
        self.gpu_indices_by_server = gpu_indices_by_server
        super().__init__(timeout=timeout)

    def _init_status_structures(self) -> None:
        """Initialize GPU status data structures."""
        gpus_by_server = {}

        if self.gpu_indices_by_server is None:
            # Handle localhost case - get physical GPU index
            assert torch.cuda.is_available(), "CUDA must be available when gpu_indices_by_server is None"
            device_index = torch.cuda.current_device()
            cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            if cuda_visible_devices:
                visible_devices = [int(d.strip()) for d in cuda_visible_devices.split(',')]
                physical_device_index = visible_devices[device_index]
            else:
                physical_device_index = device_index

            gpu_status = GPUStatus(
                server='localhost',
                index=physical_device_index,
                max_memory=0,
                processes=[],
                window_size=10,  # Default window size
                memory_window=[],
                util_window=[],
                memory_stats={},
                util_stats={},
                connected=False,
            )
            gpus_by_server['localhost'] = [gpu_status]
        else:
            # Regular dictionary of server -> indices
            for server, indices in self.gpu_indices_by_server.items():
                server_gpus = []
                for gpu_idx in indices:
                    gpu_status = GPUStatus(
                        server=server,
                        index=gpu_idx,
                        max_memory=0,
                        processes=[],
                        window_size=10,  # Default window size
                        memory_window=[],
                        util_window=[],
                        memory_stats={},
                        util_stats={},
                        connected=False,
                    )
                    server_gpus.append(gpu_status)
                gpus_by_server[server] = server_gpus

        self.gpus_by_server = gpus_by_server

    def _get_servers_list(self) -> List[str]:
        """Get list of servers being monitored."""
        return list(self.gpus_by_server.keys())

    def _update_single_server(self, server: str) -> None:
        """Update all GPUs on a single server using batched queries"""
        gpu_indices = [gpu.index for gpu in self.gpus_by_server[server]]

        # Get batched GPU info for all GPUs on this server
        server_gpus_info = self._get_server_gpus_info(server, gpu_indices, self.ssh_pool, timeout=self.timeout)

        # Update each GPU with the batched results
        for gpu in self.gpus_by_server[server]:
            self._update_single_gpu(gpu, server_gpus_info[gpu.index])

    def _update_single_gpu(self, gpu_status: GPUStatus, gpu_info: Dict[str, Any]) -> None:
        """Update a single GPU with the provided info"""
        # Update connection status
        gpu_status.connected = gpu_info['connected']

        # If query failed, set all fields to None/empty except server and index
        if not gpu_info['connected']:
            gpu_status.max_memory = 0
            gpu_status.processes = []
            gpu_status.memory_window = []
            gpu_status.util_window = []
            gpu_status.memory_stats = {}
            gpu_status.util_stats = {}
            return

        # Update processes
        gpu_status.processes = gpu_info['processes'] or []

        # Update max memory
        gpu_status.max_memory = gpu_info['max_memory'] or 0

        # Initialize windows if they are empty (GPU just reconnected)
        if not gpu_status.memory_window:
            gpu_status.memory_window = []
        if not gpu_status.util_window:
            gpu_status.util_window = []

        # Update rolling windows
        gpu_status.memory_window.append(gpu_info['current_memory'])
        gpu_status.util_window.append(gpu_info['current_util'])

        # Trim windows if needed
        if len(gpu_status.memory_window) > gpu_status.window_size:
            gpu_status.memory_window = gpu_status.memory_window[-gpu_status.window_size:]
        if len(gpu_status.util_window) > gpu_status.window_size:
            gpu_status.util_window = gpu_status.util_window[-gpu_status.window_size:]

        # Update stats
        gpu_status.memory_stats = {
            'min': min(gpu_status.memory_window),
            'max': max(gpu_status.memory_window),
            'avg': sum(gpu_status.memory_window) / len(gpu_status.memory_window),
        }
        gpu_status.util_stats = {
            'min': min(gpu_status.util_window),
            'max': max(gpu_status.util_window),
            'avg': sum(gpu_status.util_window) / len(gpu_status.util_window),
        }

    def _check(self) -> Dict:
        """Returns current status of all GPUs without rolling windows"""
        return {
            f"{gpu.server}:{gpu.index}": {
                'server': gpu.server,
                'index': gpu.index,
                'max_memory': gpu.max_memory,
                'memory_min': gpu.memory_stats.get('min') if gpu.memory_stats else None,
                'memory_max': gpu.memory_stats.get('max') if gpu.memory_stats else None,
                'memory_avg': gpu.memory_stats.get('avg') if gpu.memory_stats else None,
                'util_min': gpu.util_stats.get('min') if gpu.util_stats else None,
                'util_max': gpu.util_stats.get('max') if gpu.util_stats else None,
                'util_avg': gpu.util_stats.get('avg') if gpu.util_stats else None,
                'connected': gpu.connected,
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
            if gpu.connected
        ]

    @property
    def disconnected_gpus(self) -> Dict[str, List[int]]:
        """Get all disconnected GPUs"""
        result = {}
        for server, server_gpus in self.gpus_by_server.items():
            disconnected = [gpu.index for gpu in server_gpus if not gpu.connected]
            if len(disconnected) > 0:
                result[server] = disconnected
        return result

    def get_all_running_commands(self) -> List[str]:
        """Get all running commands on all servers"""
        all_gpus = [
            gpu for server_gpus in self.gpus_by_server.values() for gpu in server_gpus
            if gpu.connected
        ]
        all_processes = [
            process for gpu in all_gpus for process in gpu.processes
            if process.cmd.startswith('python main.py --config-filepath')
        ]
        all_running_commands = [process.cmd for process in all_processes]
        return all_running_commands

    @staticmethod
    def _get_server_gpus_mem_util(
        server: str,
        gpu_indices: List[int],
        pool: SSHConnectionPool,
    ) -> Dict[int, Dict[str, int]]:
        if not gpu_indices:
            return {}

        gpu_list = ','.join(map(str, gpu_indices))
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits',
            f'--id={gpu_list}',
        ]
        output = pool.execute(server, cmd)

        results: Dict[int, Dict[str, int]] = {}
        for line in output.splitlines():
            parts = line.split(', ')
            if len(parts) == 4:
                gpu_idx = int(parts[0])
                max_memory = int(parts[1])
                memory_used = int(parts[2])
                gpu_util = int(parts[3])
                results[gpu_idx] = {
                    'memory': memory_used,
                    'util': gpu_util,
                    'max_memory': max_memory,
                }
        assert results.keys() == set(gpu_indices), f"{results.keys()=}, {gpu_indices=}"
        return results

    @staticmethod
    def _get_server_gpus_processes(
        server: str,
        gpu_indices: List[int],
        pool: SSHConnectionPool,
    ) -> Dict[int, List[ProcessInfo]]:
        if not gpu_indices:
            return {}

        gpu_list = ','.join(map(str, gpu_indices))
        uuid_cmd = ['nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader', f'--id={gpu_list}']
        uuid_output = pool.execute(server, uuid_cmd)

        index_to_uuid: Dict[int, str] = {}
        for line in uuid_output.splitlines():
            parts = line.split(', ')
            if len(parts) == 2:
                gpu_idx = int(parts[0])
                gpu_uuid = parts[1]
                index_to_uuid[gpu_idx] = gpu_uuid

        pids_cmd = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader']
        pids_output = pool.execute(server, pids_cmd)

        uuid_to_pids: Dict[str, List[str]] = {}
        for line in pids_output.splitlines():
            parts = line.split(', ')
            if len(parts) == 2:
                uuid, pid = parts[0], parts[1]
                uuid_to_pids.setdefault(uuid, []).append(pid)

        all_processes = get_all_processes(server, pool)

        results: Dict[int, List[ProcessInfo]] = {}
        for gpu_idx in gpu_indices:
            if gpu_idx in index_to_uuid:
                uuid = index_to_uuid[gpu_idx]
                pids = uuid_to_pids.get(uuid, [])
                processes = [all_processes[pid] for pid in pids if pid in all_processes]
                results[gpu_idx] = processes
            else:
                results[gpu_idx] = []
        assert results.keys() == set(gpu_indices), f"{results.keys()=}, {gpu_indices=}"
        return results

    def _get_server_gpus_info(
        self,
        server: str,
        gpu_indices: List[int],
        pool: SSHConnectionPool,
        timeout: int = 10,
    ) -> Dict[int, Dict[str, Any]]:
        @with_timeout(seconds=timeout)
        def _query():
            results: Dict[int, Dict[str, Any]] = {}
            gpu_mem_util = self._get_server_gpus_mem_util(server, gpu_indices, pool)
            gpu_processes = self._get_server_gpus_processes(server, gpu_indices, pool)
            for gpu_idx in gpu_indices:
                results[gpu_idx] = {
                    'server': server,
                    'index': gpu_idx,
                    'max_memory': gpu_mem_util[gpu_idx]['max_memory'],
                    'current_memory': gpu_mem_util[gpu_idx]['memory'],
                    'current_util': gpu_mem_util[gpu_idx]['util'],
                    'processes': gpu_processes[gpu_idx],
                    'connected': True,
                }
            return results

        try:
            return _query()
        except Exception as exc:  # noqa: BLE001 - log and mark disconnected
            print(f"ERROR: Failed to get GPU info for server {server}, GPUs {gpu_indices}: {exc}")
            return {
                gpu_idx: {
                    'server': server,
                    'index': gpu_idx,
                    'max_memory': None,
                    'current_memory': None,
                    'current_util': None,
                    'processes': [],
                    'connected': False,
                }
                for gpu_idx in gpu_indices
            }
