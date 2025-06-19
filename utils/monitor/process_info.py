from typing import Dict, TypedDict
from utils.automation.ssh_utils import get_ssh_cmd, safe_check_output


class ProcessInfo(TypedDict):
    pid: str
    user: str
    cmd: str
    start_time: str


def get_all_processes(server: str) -> Dict[str, ProcessInfo]:
    """Get information for all processes on a server"""
    cmd = ["ps", "-eo", "pid=,user=,lstart=,cmd="]
    cmd = get_ssh_cmd(server, cmd)
    result = safe_check_output(cmd, server, "process list query")
    if result is None:
        return {}
    
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
