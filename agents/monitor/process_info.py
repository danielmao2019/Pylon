from typing import Dict
from dataclasses import dataclass, asdict
from agents.connector.pool import SSHConnectionPool


@dataclass
class ProcessInfo:
    """Information about a running process."""
    pid: str
    user: str
    cmd: str
    start_time: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def get_all_processes(server: str, pool: SSHConnectionPool) -> Dict[str, ProcessInfo]:
    """Get information for all processes on a server"""
    cmd = ["ps", "-eo", "pid=,user=,lstart=,cmd="]
    result = pool.execute(server, cmd)

    lines = result.splitlines()
    result_dict = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 7:  # Ensure we have enough parts
            result_dict[parts[0]] = ProcessInfo(
                pid=parts[0],
                user=parts[1],
                start_time=' '.join(parts[2:7]),
                cmd=' '.join(parts[7:])
            )
    return result_dict
