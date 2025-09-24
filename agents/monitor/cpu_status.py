from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
from agents.monitor.process_info import ProcessInfo, get_all_processes
from utils.ssh.pool import SSHConnectionPool
from utils.timeout import with_timeout


@dataclass
class CPUStatus:
    """Status information for a CPU/server."""
    server: str
    max_memory: int  # Total system memory in MB
    cpu_cores: int  # Number of CPU cores
    processes: List[ProcessInfo] = field(default_factory=list)
    window_size: int = 0
    memory_window: List[int] = field(default_factory=list)  # Memory usage in MB
    cpu_window: List[float] = field(default_factory=list)  # CPU utilization percentage
    load_window: List[float] = field(default_factory=list)  # Load average (1min)
    memory_stats: Dict[str, Optional[float]] = field(default_factory=dict)
    cpu_stats: Dict[str, Optional[float]] = field(default_factory=dict)
    load_stats: Dict[str, Optional[float]] = field(default_factory=dict)
    connected: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def get_server_cpu_mem_util(server: str, pool: SSHConnectionPool) -> Dict[str, Any]:
    """Get memory and CPU utilization for a server.

    Args:
        server: The server to query
        pool: SSH connection pool instance

    Returns:
        Dict with 'memory_total', 'memory_used', 'cpu_util', and 'load_avg' keys
    """
    # Get memory info from /proc/meminfo
    mem_cmd = ["cat", "/proc/meminfo"]
    mem_output = pool.execute(server, mem_cmd)

    mem_total = 0
    mem_available = 0

    for line in mem_output.splitlines():
        if line.startswith('MemTotal:'):
            mem_total = int(line.split()[1]) // 1024  # Convert KB to MB
        elif line.startswith('MemAvailable:'):
            mem_available = int(line.split()[1]) // 1024  # Convert KB to MB

    mem_used = mem_total - mem_available

    # Get CPU utilization using top (without shell pipes for localhost compatibility)
    top_cmd = ["top", "-bn1"]
    top_output = pool.execute(server, top_cmd)
    
    cpu_util = None  # Initialize to None to indicate parsing failure
    for line in top_output.splitlines():
        if '%Cpu' in line:
            # Parse line like: %Cpu(s):  5.9 us,  1.2 sy,  0.0 ni, 92.6 id,  0.3 wa,  0.0 hi,  0.0 si,  0.0 st
            parts = line.split()
            for i, part in enumerate(parts):
                if 'us,' in part and i > 0:  # Find 'us,' and ensure there's a previous part
                    try:
                        cpu_util = float(parts[i-1])  # Get the number from the previous part
                    except ValueError:
                        cpu_util = None  # Keep as None on parsing failure
                    break
            break

    # Get load average
    load_cmd = ["cat", "/proc/loadavg"]
    load_output = pool.execute(server, load_cmd)
    load_avg = float(load_output.split()[0])  # 1-minute load average

    # Get number of CPU cores
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


def get_server_cpu_processes(server: str, pool: SSHConnectionPool) -> List[ProcessInfo]:
    """Get list of processes running on a server.

    Args:
        server: The server to query
        pool: SSH connection pool instance

    Returns:
        List of ProcessInfo objects
    """
    # Get all processes
    all_processes = get_all_processes(server, pool)

    # Return all processes (can be filtered later if needed)
    return list(all_processes.values())


def get_server_cpu_info(server: str, pool: SSHConnectionPool, timeout: int = 10) -> Dict[str, Any]:
    """Get CPU information for a server.

    Args:
        server: The server to query
        pool: SSH connection pool instance
        timeout: Timeout in seconds for the query (default: 10)

    Returns:
        Dict with CPU info
    """
    assert isinstance(server, str), f"{server=}"
    assert isinstance(pool, SSHConnectionPool), f"{pool=}"
    assert isinstance(timeout, int), f"{timeout=}"

    @with_timeout(seconds=timeout)
    def _get_server_cpu_info():
        # Get memory and CPU utilization
        cpu_stats = get_server_cpu_mem_util(server, pool)

        # Get process info
        processes = get_server_cpu_processes(server, pool)

        return {
            'server': server,
            'max_memory': cpu_stats['memory_total'],
            'current_memory': cpu_stats['memory_used'],
            'current_cpu': cpu_stats['cpu_util'],
            'current_load': cpu_stats['load_avg'],
            'cpu_cores': cpu_stats['cpu_cores'],
            'processes': processes,
            'connected': True,
        }

    try:
        return _get_server_cpu_info()
    except Exception as e:
        print(f"ERROR: Failed to get CPU info for server {server}: {e}")
        return {
            'server': server,
            'max_memory': None,
            'current_memory': None,
            'current_cpu': None,
            'current_load': None,
            'cpu_cores': None,
            'processes': None,
            'connected': False,
        }
