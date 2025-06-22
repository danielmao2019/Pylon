from typing import Dict, TypedDict
from utils.ssh import SSHConnectionPool


class ProcessInfo(TypedDict):
    pid: str
    user: str
    cmd: str
    start_time: str


def get_all_processes(server: str, pool: SSHConnectionPool) -> Dict[str, ProcessInfo]:
    """Get information for all processes on a server"""
    cmd = ["ps", "-eo", "pid=,user=,lstart=,cmd="]
    result = pool.execute(server, cmd)

    lines = result.splitlines()
    result_dict = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 7:  # Ensure we have enough parts
            result_dict[parts[0]] = {
                'pid': parts[0],
                'user': parts[1],
                'start_time': ' '.join(parts[2:7]),
                'cmd': ' '.join(parts[7:])
            }
    return result_dict
