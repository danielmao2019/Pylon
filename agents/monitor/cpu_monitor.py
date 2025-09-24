from typing import List, Dict, Optional, Any
from agents.monitor.base_monitor import BaseMonitor
from agents.monitor.cpu_status import CPUStatus
from agents.monitor.process_info import ProcessInfo, get_all_processes
from agents.connector.pool import SSHConnectionPool
from utils.timeout import with_timeout


class CPUMonitor(BaseMonitor[CPUStatus]):
    """Monitor CPU utilisation for a single server."""

    def __init__(self, server: str, timeout: int = 5, window_size: int = 10):
        self.server = server
        self.window_size = window_size
        self.status: CPUStatus
        super().__init__(timeout=timeout)

    def _initialize_state(self) -> None:
        self.status = CPUStatus(server=self.server, window_size=None)

    def _update_resource(self) -> None:
        info = self._collect_cpu_info(self.server, self.ssh_pool, timeout=self.timeout)

        if not info['connected']:
            self._mark_disconnected()
            return

        # Ensure previous status state exists
        status = self.status
        status.window_size = self.window_size
        if not status.connected:
            status.memory_window.clear()
            status.cpu_window.clear()
            status.load_window.clear()

        status.connected = True
        status.max_memory = info['max_memory']
        status.cpu_cores = info['cpu_cores']
        status.processes = info['processes'] or []

        current_memory = info['current_memory']
        if current_memory is not None:
            status.memory_window.append(current_memory)

        status.cpu_window.append(info['current_cpu'])

        current_load = info['current_load']
        if current_load is not None:
            status.load_window.append(current_load)

        self._trim_windows(status)
        self._update_stats(status)

    def _mark_disconnected(self) -> None:
        self.status = CPUStatus(server=self.server, connected=False, window_size=None)
        assert self.status.window_size is None

    @staticmethod
    def _trim_windows(status: CPUStatus) -> None:
        assert status.window_size is not None, "window_size must be set when CPU is connected"
        if len(status.memory_window) > status.window_size:
            status.memory_window = status.memory_window[-status.window_size:]
        if len(status.cpu_window) > status.window_size:
            status.cpu_window = status.cpu_window[-status.window_size:]
        if len(status.load_window) > status.window_size:
            status.load_window = status.load_window[-status.window_size:]

    @staticmethod
    def _update_stats(status: CPUStatus) -> None:
        memory_values = status.memory_window
        if memory_values:
            status.memory_stats = {
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': sum(memory_values) / len(memory_values),
            }
        else:
            status.memory_stats = {'min': None, 'max': None, 'avg': None}

        valid_cpu_values = [value for value in status.cpu_window if value is not None]
        if valid_cpu_values:
            status.cpu_stats = {
                'min': min(valid_cpu_values),
                'max': max(valid_cpu_values),
                'avg': sum(valid_cpu_values) / len(valid_cpu_values),
            }
        else:
            status.cpu_stats = {'min': None, 'max': None, 'avg': None}

        load_values = status.load_window
        if load_values:
            status.load_stats = {
                'min': min(load_values),
                'max': max(load_values),
                'avg': sum(load_values) / len(load_values),
            }
        else:
            status.load_stats = {'min': None, 'max': None, 'avg': None}

    @property
    def cpu(self) -> CPUStatus:
        return self.status

    def get_all_running_commands(self) -> List[str]:
        return [
            process.cmd
            for process in self.status.processes
            if process.cmd.startswith('python main.py --config-filepath')
        ]

    @staticmethod
    def _collect_cpu_mem_util(server: str, pool: SSHConnectionPool) -> Dict[str, Any]:
        mem_cmd = ["cat", "/proc/meminfo"]
        mem_output = pool.execute(server, mem_cmd)

        mem_total = 0
        mem_available = 0
        for line in mem_output.splitlines():
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1]) // 1024
            elif line.startswith('MemAvailable:'):
                mem_available = int(line.split()[1]) // 1024

        mem_used = mem_total - mem_available

        top_cmd = ["top", "-bn1"]
        top_output = pool.execute(server, top_cmd)
        cpu_util: Optional[float] = None
        for line in top_output.splitlines():
            if '%Cpu' in line:
                parts = line.split()
                for idx, part in enumerate(parts):
                    if 'us,' in part and idx > 0:
                        try:
                            cpu_util = float(parts[idx - 1])
                        except ValueError:
                            cpu_util = None
                        break
                break

        load_cmd = ["cat", "/proc/loadavg"]
        load_output = pool.execute(server, load_cmd)
        load_avg = float(load_output.split()[0])

        cores_cmd = ["nproc"]
        cores_output = pool.execute(server, cores_cmd)
        cpu_cores = int(cores_output.strip())

        return {
            'memory_total': mem_total,
            'memory_used': mem_used,
            'cpu_util': cpu_util,
            'load_avg': load_avg,
            'cpu_cores': cpu_cores,
        }

    @staticmethod
    def _collect_cpu_processes(server: str, pool: SSHConnectionPool) -> List[ProcessInfo]:
        all_processes = get_all_processes(server, pool)
        return list(all_processes.values())

    def _collect_cpu_info(
        self,
        server: str,
        pool: SSHConnectionPool,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        @with_timeout(seconds=timeout)
        def _query():
            stats = self._collect_cpu_mem_util(server, pool)
            processes = self._collect_cpu_processes(server, pool)
            return {
                'server': server,
                'max_memory': stats['memory_total'],
                'current_memory': stats['memory_used'],
                'current_cpu': stats['cpu_util'],
                'current_load': stats['load_avg'],
                'cpu_cores': stats['cpu_cores'],
                'processes': processes,
                'connected': True,
            }

        try:
            return _query()
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: Failed to get CPU info for server {server}: {exc}")
            return {
                'server': server,
                'max_memory': None,
                'current_memory': None,
                'current_cpu': None,
                'current_load': None,
                'cpu_cores': None,
                'processes': [],
                'connected': False,
            }
