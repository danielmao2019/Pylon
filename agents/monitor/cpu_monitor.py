from typing import List, Dict, Optional, Any
from agents.monitor.base_monitor import BaseMonitor
from agents.monitor.cpu_status import CPUStatus, get_server_cpu_info


class CPUMonitor(BaseMonitor[CPUStatus]):

    def __init__(self, servers: Optional[List[str]] = None, timeout: int = 5):
        """
        Initialize CPU monitor with servers to monitor.

        Args:
            servers: List of server names to monitor, or None for localhost only
            timeout: SSH command timeout in seconds
        """
        self.servers_list = servers
        super().__init__(timeout=timeout)

    def _init_status_structures(self) -> None:
        """Initialize CPU status data structures."""
        if self.servers_list is None:
            self.cpus_by_server: Dict[str, CPUStatus] = {
                'localhost': CPUStatus(server='localhost', connected=False)
            }
        else:
            self.cpus_by_server = {
                server: CPUStatus(server=server, connected=False)
                for server in self.servers_list
            }

    def _get_servers_list(self) -> List[str]:
        """Get list of servers being monitored."""
        return list(self.cpus_by_server.keys())

    def _update_single_server(self, server: str) -> None:
        """Update CPU info for a single server"""
        # Get CPU info for this server
        server_cpu_info = get_server_cpu_info(server, self.ssh_pool, timeout=self.timeout)

        # Update the CPU with the results
        self._update_single_cpu(self.cpus_by_server[server], server_cpu_info)

    def _update_single_cpu(self, cpu_status: CPUStatus, cpu_info: Dict[str, Any]) -> None:
        """Update a single CPU with the provided info"""
        # Update connection status
        cpu_status.connected = bool(cpu_info['connected'])

        # If query failed, set all fields to None except server
        if not cpu_status.connected:
            cpu_status.max_memory = None
            cpu_status.cpu_cores = None
            cpu_status.processes = []
            cpu_status.memory_window.clear()
            cpu_status.cpu_window.clear()
            cpu_status.load_window.clear()
            cpu_status.memory_stats = None
            cpu_status.cpu_stats = None
            cpu_status.load_stats = None
            return

        # Update processes
        cpu_status.processes = cpu_info.get('processes') or []

        # Update max memory and cpu cores
        cpu_status.max_memory = cpu_info.get('max_memory')
        cpu_status.cpu_cores = cpu_info.get('cpu_cores')

        # Update rolling windows
        current_memory = cpu_info.get('current_memory')
        if current_memory is not None:
            cpu_status.memory_window.append(current_memory)

        current_cpu = cpu_info.get('current_cpu')
        cpu_status.cpu_window.append(current_cpu)

        current_load = cpu_info.get('current_load')
        if current_load is not None:
            cpu_status.load_window.append(current_load)

        # Trim windows if needed
        if len(cpu_status.memory_window) > cpu_status.window_size:
            cpu_status.memory_window = cpu_status.memory_window[-cpu_status.window_size:]
        if len(cpu_status.cpu_window) > cpu_status.window_size:
            cpu_status.cpu_window = cpu_status.cpu_window[-cpu_status.window_size:]
        if len(cpu_status.load_window) > cpu_status.window_size:
            cpu_status.load_window = cpu_status.load_window[-cpu_status.window_size:]

        # Update stats - handle None values gracefully
        memory_values = cpu_status.memory_window
        if memory_values:
            cpu_status.memory_stats = {
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': sum(memory_values) / len(memory_values),
            }
        else:
            cpu_status.memory_stats = {
                'min': None,
                'max': None,
                'avg': None,
            }

        valid_cpu_values = [v for v in cpu_status.cpu_window if v is not None]
        if valid_cpu_values:
            cpu_status.cpu_stats = {
                'min': min(valid_cpu_values),
                'max': max(valid_cpu_values),
                'avg': sum(valid_cpu_values) / len(valid_cpu_values),
            }
        else:
            cpu_status.cpu_stats = {
                'min': None,
                'max': None,
                'avg': None,
            }
        
        if cpu_status.load_window:
            cpu_status.load_stats = {
                'min': min(cpu_status.load_window),
                'max': max(cpu_status.load_window),
                'avg': sum(cpu_status.load_window) / len(cpu_status.load_window),
            }
        else:
            cpu_status.load_stats = {
                'min': None,
                'max': None,
                'avg': None,
            }

    def _check(self) -> Dict:
        """Returns current status of all CPUs without rolling windows"""
        return {
            cpu.server: {
                'server': cpu.server,
                'max_memory': cpu.max_memory,
                'memory_min': cpu.memory_stats.get('min') if cpu.memory_stats else None,
                'memory_max': cpu.memory_stats.get('max') if cpu.memory_stats else None,
                'memory_avg': cpu.memory_stats.get('avg') if cpu.memory_stats else None,
                'cpu_min': cpu.cpu_stats.get('min') if cpu.cpu_stats else None,
                'cpu_max': cpu.cpu_stats.get('max') if cpu.cpu_stats else None,
                'cpu_avg': cpu.cpu_stats.get('avg') if cpu.cpu_stats else None,
                'load_min': cpu.load_stats.get('min') if cpu.load_stats else None,
                'load_max': cpu.load_stats.get('max') if cpu.load_stats else None,
                'load_avg': cpu.load_stats.get('avg') if cpu.load_stats else None,
                'connected': cpu.connected,
            }
            for cpu in self.cpus_by_server.values()
        }

    @property
    def cpus(self) -> List[CPUStatus]:
        """Get all CPUs"""
        return list(self.cpus_by_server.values())

    @property
    def connected_cpus(self) -> List[CPUStatus]:
        """Get all connected CPUs"""
        return [cpu for cpu in self.cpus_by_server.values() if cpu.connected]

    @property
    def disconnected_cpus(self) -> List[str]:
        """Get all disconnected CPUs"""
        return [cpu.server for cpu in self.cpus_by_server.values() if not cpu.connected]

    def get_all_running_commands(self) -> List[str]:
        """Get all running commands on all servers"""
        all_cpus = [cpu for cpu in self.cpus_by_server.values() if cpu.connected]
        all_processes = [
            process for cpu in all_cpus for process in cpu.processes
            if process.cmd.startswith('python main.py --config-filepath')
        ]
        all_running_commands = [process.cmd for process in all_processes]
        return all_running_commands
