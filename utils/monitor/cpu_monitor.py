from typing import List, Dict, Optional, Any
import threading
from concurrent.futures import ThreadPoolExecutor
from utils.monitor.cpu_status import CPUStatus, get_server_cpu_info
from utils.ssh.pool import _ssh_pool


class CPUMonitor:

    def __init__(self, servers: Optional[List[str]] = None, timeout: int = 5):
        """
        Initialize CPU monitor with servers to monitor.

        Args:
            servers: List of server names to monitor, or None for localhost only
            timeout: SSH command timeout in seconds
        """
        self.cpus_by_server = self._init_cpu_status(servers)
        self.timeout = timeout
        self.servers = list(self.cpus_by_server.keys())
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Do one update first
        self.ssh_pool = _ssh_pool
        self._update()

    def _init_cpu_status(self, servers: Optional[List[str]]) -> Dict[str, CPUStatus]:
        """Initialize CPUStatus objects from server list.

        Args:
            servers: List of server names to monitor, or None for localhost only

        Returns:
            Dictionary mapping server names to CPUStatus objects
        """
        cpus_by_server = {}

        if servers is None:
            # Handle localhost case
            cpu_status: CPUStatus = {
                'server': 'localhost',
                'max_memory': None,
                'cpu_cores': None,
                'processes': None,
                'window_size': 10,  # Default window size
                'memory_window': None,
                'cpu_window': None,
                'load_window': None,
                'memory_stats': None,
                'cpu_stats': None,
                'load_stats': None,
                'connected': False,
            }
            cpus_by_server['localhost'] = cpu_status
        else:
            # Multiple servers
            for server in servers:
                cpu_status: CPUStatus = {
                    'server': server,
                    'max_memory': None,
                    'cpu_cores': None,
                    'processes': None,
                    'window_size': 10,  # Default window size
                    'memory_window': None,
                    'cpu_window': None,
                    'load_window': None,
                    'memory_stats': None,
                    'cpu_stats': None,
                    'load_stats': None,
                    'connected': False,
                }
                cpus_by_server[server] = cpu_status

        return cpus_by_server

    def start(self):
        """Starts background monitoring thread that continuously updates CPU info"""
        def monitor_loop():
            while not self._stop_event.is_set():
                self._update()

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stops background monitoring thread"""
        if self._stop_event is not None:
            self._stop_event.set()
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=1.0)

    def __del__(self):
        """Automatically stop monitoring when instance is destroyed"""
        self.stop()

    def _update(self):
        """Updates information for all CPUs using batched queries per server"""
        with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
            list(executor.map(self._update_single_server, self.servers))

    def _update_single_server(self, server: str) -> None:
        """Update CPU info for a single server"""
        # Get CPU info for this server
        server_cpu_info = get_server_cpu_info(server, self.ssh_pool, timeout=self.timeout)

        # Update the CPU with the results
        self._update_single_cpu(self.cpus_by_server[server], server_cpu_info)

    def _update_single_cpu(self, cpu_status: CPUStatus, cpu_info: Dict[str, Any]) -> None:
        """Update a single CPU with the provided info"""
        # Update connection status
        cpu_status['connected'] = cpu_info['connected']

        # If query failed, set all fields to None except server
        if not cpu_info['connected']:
            cpu_status['max_memory'] = None
            cpu_status['cpu_cores'] = None
            cpu_status['processes'] = None
            cpu_status['memory_window'] = None
            cpu_status['cpu_window'] = None
            cpu_status['load_window'] = None
            cpu_status['memory_stats'] = None
            cpu_status['cpu_stats'] = None
            cpu_status['load_stats'] = None
            return

        # Update processes
        cpu_status['processes'] = cpu_info['processes']

        # Update max memory and cpu cores
        cpu_status['max_memory'] = cpu_info['max_memory']
        cpu_status['cpu_cores'] = cpu_info['cpu_cores']

        # Initialize windows if they are None (CPU just reconnected)
        if cpu_status['memory_window'] is None:
            cpu_status['memory_window'] = []
        if cpu_status['cpu_window'] is None:
            cpu_status['cpu_window'] = []
        if cpu_status['load_window'] is None:
            cpu_status['load_window'] = []

        # Update rolling windows
        cpu_status['memory_window'].append(cpu_info['current_memory'])
        cpu_status['cpu_window'].append(cpu_info['current_cpu'])
        cpu_status['load_window'].append(cpu_info['current_load'])

        # Trim windows if needed
        if len(cpu_status['memory_window']) > cpu_status['window_size']:
            cpu_status['memory_window'] = cpu_status['memory_window'][-cpu_status['window_size']:]
        if len(cpu_status['cpu_window']) > cpu_status['window_size']:
            cpu_status['cpu_window'] = cpu_status['cpu_window'][-cpu_status['window_size']:]
        if len(cpu_status['load_window']) > cpu_status['window_size']:
            cpu_status['load_window'] = cpu_status['load_window'][-cpu_status['window_size']:]

        # Update stats
        cpu_status['memory_stats'] = {
            'min': min(cpu_status['memory_window']),
            'max': max(cpu_status['memory_window']),
            'avg': sum(cpu_status['memory_window']) / len(cpu_status['memory_window']),
        }
        cpu_status['cpu_stats'] = {
            'min': min(cpu_status['cpu_window']),
            'max': max(cpu_status['cpu_window']),
            'avg': sum(cpu_status['cpu_window']) / len(cpu_status['cpu_window']),
        }
        cpu_status['load_stats'] = {
            'min': min(cpu_status['load_window']),
            'max': max(cpu_status['load_window']),
            'avg': sum(cpu_status['load_window']) / len(cpu_status['load_window']),
        }

    def _check(self) -> Dict:
        """Returns current status of all CPUs without rolling windows"""
        return {
            cpu['server']: {
                'server': cpu['server'],
                'max_memory': cpu['max_memory'],
                'memory_min': cpu['memory_stats']['min'] if cpu['memory_stats'] is not None else None,
                'memory_max': cpu['memory_stats']['max'] if cpu['memory_stats'] is not None else None,
                'memory_avg': cpu['memory_stats']['avg'] if cpu['memory_stats'] is not None else None,
                'cpu_min': cpu['cpu_stats']['min'] if cpu['cpu_stats'] is not None else None,
                'cpu_max': cpu['cpu_stats']['max'] if cpu['cpu_stats'] is not None else None,
                'cpu_avg': cpu['cpu_stats']['avg'] if cpu['cpu_stats'] is not None else None,
                'load_min': cpu['load_stats']['min'] if cpu['load_stats'] is not None else None,
                'load_max': cpu['load_stats']['max'] if cpu['load_stats'] is not None else None,
                'load_avg': cpu['load_stats']['avg'] if cpu['load_stats'] is not None else None,
                'connected': cpu['connected'],
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
        return [
            cpu for cpu in self.cpus_by_server.values()
            if cpu['connected']
        ]

    @property
    def disconnected_cpus(self) -> List[str]:
        """Get all disconnected CPUs"""
        return [
            cpu['server'] for cpu in self.cpus_by_server.values()
            if not cpu['connected']
        ]

    def get_all_running_commands(self) -> List[str]:
        """Get all running commands on all servers"""
        all_cpus = [
            cpu for cpu in self.cpus_by_server.values()
            if cpu['connected']
        ]
        all_processes = [
            process for cpu in all_cpus for process in cpu['processes']
            if process['cmd'].startswith('python main.py --config-filepath')
        ]
        all_running_commands = [process['cmd'] for process in all_processes]
        return all_running_commands

    def log_stats(self, logger):
        """Logs status of all monitored CPUs"""
        stats = self._check()
        assert len(stats) == 1, "Only support single server monitoring for now."
        stats = list(stats.values())[0]
        logger.update_buffer(stats)
