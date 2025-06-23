from typing import List, Dict, TypedDict, Optional, Any
from utils.monitor.process_info import ProcessInfo, get_all_processes
from utils.ssh.pool import SSHConnectionPool
from utils.timeout import with_timeout


class CPUStatus(TypedDict):
    server: str
    max_memory: int  # Total system memory in MB
    processes: List[ProcessInfo]
    window_size: int
    memory_window: List[int]  # Memory usage in MB
    cpu_window: List[float]  # CPU utilization percentage
    load_window: List[float]  # Load average (1min)
    memory_stats: dict[str, Optional[float]]
    cpu_stats: dict[str, Optional[float]]
    load_stats: dict[str, Optional[float]]
    connected: bool


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

    # Get CPU utilization using top
    cpu_cmd = ["top", "-bn1", "|", "grep", "'^%Cpu'", "|", "awk", "'{print $2}'", "|", "sed", "'s/%us,//'"]
    cpu_output = pool.execute(server, cpu_cmd)

    try:
        cpu_util = float(cpu_output.strip())
    except ValueError:
        # Fallback: parse top output more carefully
        top_cmd = ["top", "-bn1", "|", "head", "-n", "10"]
        top_output = pool.execute(server, top_cmd)
        cpu_util = 0.0
        for line in top_output.splitlines():
            if '%Cpu' in line:
                # Parse line like: %Cpu(s):  5.9 us,  1.2 sy,  0.0 ni, 92.6 id,  0.3 wa,  0.0 hi,  0.0 si,  0.0 st
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'us,' in part:
                        cpu_util = float(part.replace('us,', ''))
                        break
                break

    # Get load average
    load_cmd = ["cat", "/proc/loadavg"]
    load_output = pool.execute(server, load_cmd)
    load_avg = float(load_output.split()[0])  # 1-minute load average

    return {
        'memory_total': mem_total,
        'memory_used': mem_used,
        'cpu_util': cpu_util,
        'load_avg': load_avg
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
            'processes': None,
            'connected': False,
        }
