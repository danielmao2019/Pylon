from typing import Dict, Optional, Any, List
import os
import torch
from agents.monitor.base_monitor import BaseMonitor
from agents.monitor.gpu_status import GPUStatus
from agents.monitor.process_info import ProcessInfo, get_all_processes
from agents.connector.pool import SSHConnectionPool
from utils.timeout import with_timeout


class GPUMonitor(BaseMonitor[GPUStatus]):
    """Monitor utilisation for a single GPU on a server."""

    def __init__(self, server: str, index: int, timeout: int = 5):
        self.server = server
        self.index = index
        self.status: GPUStatus
        super().__init__(timeout=timeout)

    def _initialize_state(self) -> None:
        if self.server == 'localhost' and self.index is None:
            assert torch.cuda.is_available(), "CUDA must be available when monitoring localhost GPU"
            device_index = torch.cuda.current_device()
            cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            if cuda_visible_devices:
                visible_devices = [int(d.strip()) for d in cuda_visible_devices.split(',')]
                physical_device_index = visible_devices[device_index]
            else:
                physical_device_index = device_index
            self.index = physical_device_index
        self.status = GPUStatus(server=self.server, index=self.index, connected=False, window_size=10)

    def _update_resource(self) -> None:
        info = self._collect_gpu_info(self.server, self.index, self.ssh_pool, timeout=self.timeout)

        if not info['connected']:
            self.status = GPUStatus(server=self.server, index=self.index, connected=False, window_size=self.status.window_size)
            return

        status = self.status
        if not status.connected:
            status.memory_window.clear()
            status.util_window.clear()

        status.connected = True
        status.max_memory = info['max_memory'] or 0
        status.processes = info['processes'] or []

        status.memory_window.append(info['current_memory'])
        status.util_window.append(info['current_util'])

        if len(status.memory_window) > status.window_size:
            status.memory_window = status.memory_window[-status.window_size:]
        if len(status.util_window) > status.window_size:
            status.util_window = status.util_window[-status.window_size:]

        status.memory_stats = {
            'min': min(status.memory_window),
            'max': max(status.memory_window),
            'avg': sum(status.memory_window) / len(status.memory_window),
        }
        status.util_stats = {
            'min': min(status.util_window),
            'max': max(status.util_window),
            'avg': sum(status.util_window) / len(status.util_window),
        }

    @property
    def gpu(self) -> GPUStatus:
        return self.status

    @property
    def connected(self) -> bool:
        return self.status.connected

    def get_all_running_commands(self) -> List[str]:
        return [
            process.cmd
            for process in self.status.processes
            if process.cmd.startswith('python main.py --config-filepath')
        ]

    @staticmethod
    def _collect_gpu_mem_util(
        server: str,
        index: int,
        pool: SSHConnectionPool,
    ) -> Dict[str, int]:
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits',
            f'--id={index}',
        ]
        output = pool.execute(server, cmd)
        parts = output.split(', ')
        if len(parts) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi output: {output}")
        return {
            'max_memory': int(parts[1]),
            'memory': int(parts[2]),
            'util': int(parts[3]),
        }

    @staticmethod
    def _collect_gpu_processes(
        server: str,
        index: int,
        pool: SSHConnectionPool,
    ) -> List[ProcessInfo]:
        uuid_cmd = ['nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader', f'--id={index}']
        uuid_output = pool.execute(server, uuid_cmd).strip()
        parts = uuid_output.split(', ')
        if len(parts) != 2:
            return []
        gpu_uuid = parts[1]

        pids_cmd = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader']
        pids_output = pool.execute(server, pids_cmd)

        pids: List[str] = []
        for line in pids_output.splitlines():
            uuid, pid = line.split(', ')
            if uuid == gpu_uuid:
                pids.append(pid)

        all_processes = get_all_processes(server, pool)
        return [all_processes[pid] for pid in pids if pid in all_processes]

    def _collect_gpu_info(
        self,
        server: str,
        index: int,
        pool: SSHConnectionPool,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        @with_timeout(seconds=timeout)
        def _query():
            mem_util = self._collect_gpu_mem_util(server, index, pool)
            processes = self._collect_gpu_processes(server, index, pool)
            return {
                'server': server,
                'index': index,
                'max_memory': mem_util['max_memory'],
                'current_memory': mem_util['memory'],
                'current_util': mem_util['util'],
                'processes': processes,
                'connected': True,
            }

        try:
            return _query()
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: Failed to get GPU info for server {server}, GPU {index}: {exc}")
            return {
                'server': server,
                'index': index,
                'max_memory': None,
                'current_memory': None,
                'current_util': None,
                'processes': [],
                'connected': False,
            }
