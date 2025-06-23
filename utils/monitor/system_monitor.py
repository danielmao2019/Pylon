from typing import List, Dict, Optional, Any
from utils.monitor.gpu_monitor import GPUMonitor
from utils.monitor.cpu_monitor import CPUMonitor
from utils.monitor.gpu_status import GPUStatus
from utils.monitor.cpu_status import CPUStatus


class SystemMonitor:
    """
    Combined monitor that aggregates both CPU and GPU information.
    Provides a unified interface for system resource monitoring.
    """

    def __init__(
        self,
        gpu_indices_by_server: Optional[Dict[str, List[int]]] = None,
        cpu_servers: Optional[List[str]] = None,
        timeout: int = 5
    ):
        """
        Initialize system monitor with both CPU and GPU monitoring.

        Args:
            gpu_indices_by_server: Dictionary mapping server names to lists of GPU indices, or None for localhost
            cpu_servers: List of server names to monitor for CPU, or None for localhost only
            timeout: SSH command timeout in seconds
        """
        self.timeout = timeout

        # Initialize GPU monitor
        self.gpu_monitor = GPUMonitor(gpu_indices_by_server, timeout=timeout)

        # Initialize CPU monitor
        # If cpu_servers is not provided, use the same servers as GPU monitor
        if cpu_servers is None and gpu_indices_by_server is not None:
            cpu_servers = list(gpu_indices_by_server.keys())
        self.cpu_monitor = CPUMonitor(cpu_servers, timeout=timeout)

        # Track if monitoring is started
        self._monitoring_started = False

    def start(self):
        """Start both CPU and GPU monitoring"""
        self.gpu_monitor.start()
        self.cpu_monitor.start()
        self._monitoring_started = True

    def stop(self):
        """Stop both CPU and GPU monitoring"""
        self.gpu_monitor.stop()
        self.cpu_monitor.stop()
        self._monitoring_started = False

    def __del__(self):
        """Automatically stop monitoring when instance is destroyed"""
        self.stop()

    @property
    def gpus(self) -> List[GPUStatus]:
        """Get all GPUs"""
        return self.gpu_monitor.gpus

    @property
    def cpus(self) -> List[CPUStatus]:
        """Get all CPUs"""
        return self.cpu_monitor.cpus

    @property
    def connected_gpus(self) -> List[GPUStatus]:
        """Get all connected GPUs"""
        return self.gpu_monitor.connected_gpus

    @property
    def connected_cpus(self) -> List[CPUStatus]:
        """Get all connected CPUs"""
        return self.cpu_monitor.connected_cpus

    @property
    def disconnected_gpus(self) -> Dict[str, List[int]]:
        """Get all disconnected GPUs"""
        return self.gpu_monitor.disconnected_gpus

    @property
    def disconnected_cpus(self) -> List[str]:
        """Get all disconnected CPUs"""
        return self.cpu_monitor.disconnected_cpus

    def get_all_running_commands(self) -> List[str]:
        """Get all running commands from both CPU and GPU processes"""
        gpu_commands = self.gpu_monitor.get_all_running_commands()
        cpu_commands = self.cpu_monitor.get_all_running_commands()

        # Combine and deduplicate commands
        all_commands = list(set(gpu_commands + cpu_commands))
        return all_commands

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including both CPU and GPU info"""
        gpu_stats = self.gpu_monitor._check()
        cpu_stats = self.cpu_monitor._check()

        return {
            'gpu_stats': gpu_stats,
            'cpu_stats': cpu_stats,
            'disconnected_gpus': self.disconnected_gpus,
            'disconnected_cpus': self.disconnected_cpus,
            'monitoring_active': self._monitoring_started,
        }

    def log_stats(self, logger):
        """Log stats from both CPU and GPU monitors"""
        # Log GPU stats
        gpu_stats = self.gpu_monitor._check()
        if gpu_stats:
            # Prefix GPU stats with 'gpu_'
            gpu_stats_prefixed = {}
            for gpu_key, gpu_data in gpu_stats.items():
                for key, value in gpu_data.items():
                    gpu_stats_prefixed[f'gpu_{key}'] = value
            logger.update_buffer(gpu_stats_prefixed)

        # Log CPU stats
        cpu_stats = self.cpu_monitor._check()
        if cpu_stats:
            # Prefix CPU stats with 'cpu_'
            cpu_stats_prefixed = {}
            for cpu_key, cpu_data in cpu_stats.items():
                for key, value in cpu_data.items():
                    if key != 'server':  # Avoid duplicate server field
                        cpu_stats_prefixed[f'cpu_{key}'] = value
            logger.update_buffer(cpu_stats_prefixed)
