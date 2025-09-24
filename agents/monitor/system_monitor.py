from typing import Dict, List, Optional, Any
from agents.connector.pool import _ssh_pool
from agents.connector.error import SSHCommandError
from agents.monitor.gpu_monitor import GPUMonitor
from agents.monitor.cpu_monitor import CPUMonitor
from agents.monitor.gpu_status import GPUStatus
from agents.monitor.cpu_status import CPUStatus


class SystemMonitor:
    """Monitor CPU and GPUs for a single server."""

    def __init__(
        self,
        server: str,
        gpu_indices: Optional[List[int]] = None,
        timeout: int = 5,
    ):
        self.server = server
        self.timeout = timeout
        self.cpu_monitor = CPUMonitor(server, timeout=timeout)
        if gpu_indices is None:
            gpu_indices = self._discover_gpu_indices(server)
        self.gpu_monitors: Dict[int, GPUMonitor] = {
            idx: GPUMonitor(server, idx, timeout=timeout)
            for idx in gpu_indices
        }
        self._monitoring_started = False

    def start(self) -> None:
        self.cpu_monitor.start()
        for monitor in self.gpu_monitors.values():
            monitor.start()
        self._monitoring_started = True

    def stop(self) -> None:
        self.cpu_monitor.stop()
        for monitor in self.gpu_monitors.values():
            monitor.stop()
        self._monitoring_started = False

    def __del__(self) -> None:
        self.stop()

    @property
    def cpu(self) -> CPUStatus:
        return self.cpu_monitor.cpu

    @property
    def gpus(self) -> List[GPUStatus]:
        return [monitor.gpu for monitor in self.gpu_monitors.values()]

    @property
    def connected_cpu(self) -> Optional[CPUStatus]:
        status = self.cpu_monitor.cpu
        return status if status.connected else None

    @property
    def connected_gpus(self) -> List[GPUStatus]:
        return [gpu for gpu in self.gpus if gpu.connected]

    @property
    def disconnected_cpu(self) -> Optional[str]:
        return None if self.cpu_monitor.cpu.connected else self.server

    @property
    def disconnected_gpus(self) -> List[int]:
        return [gpu.index for gpu in self.gpus if not gpu.connected]

    def get_all_running_commands(self) -> List[str]:
        commands = self.cpu_monitor.get_all_running_commands()
        for monitor in self.gpu_monitors.values():
            commands.extend(monitor.get_all_running_commands())
        return list(set(commands))

    def get_system_status(self) -> Dict[str, Any]:
        return {
            'cpu': self.cpu_monitor.cpu.to_dict(),
            'gpus': [gpu.to_dict() for gpu in self.gpus],
            'disconnected_cpu': self.disconnected_cpu,
            'disconnected_gpus': self.disconnected_gpus,
            'monitoring_active': self._monitoring_started,
        }

    def log_stats(self, logger) -> None:
        cpu = self.cpu
        gpu_stats = {
            f'gpu_{gpu.index}_{field}': value
            for gpu in self.gpus
            for field, value in gpu.to_dict().items()
            if field != 'server'
        }
        cpu_stats = {
            f'cpu_{field}': value
            for field, value in cpu.to_dict().items()
            if field != 'server'
        }
        logger.update_buffer({**cpu_stats, **gpu_stats})

    def _discover_gpu_indices(self, server: str) -> List[int]:
        try:
            output = _ssh_pool.execute(
                server,
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            )
        except SSHCommandError:
            return []
        except Exception as exc:  # noqa: BLE001 - discovery failures are non-fatal
            print(f"WARNING: Unable to query GPUs for {server}: {exc}")
            return []

        indices: List[int] = []
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                indices.append(int(stripped.split()[0]))
            except ValueError:
                continue
        return indices
