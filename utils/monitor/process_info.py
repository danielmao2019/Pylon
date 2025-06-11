from typing import Dict, TypedDict
import subprocess


class ProcessInfo(TypedDict):
    pid: str
    user: str
    cmd: str
    start_time: str


def get_all_processes(server: str) -> Dict[str, ProcessInfo]:
    """Get information for all processes on a server"""
    cmd = ["ps", "-eo", "pid=,user=,lstart=,cmd="]
    if server != 'localhost':
        cmd = ['ssh', server] + cmd
    lines = subprocess.check_output(cmd).decode().splitlines()
    result = {}
    for line in lines:
        parts = line.strip().split()
        result[parts[0]] = {
            'pid': parts[0],
            'user': parts[1],
            'start_time': ' '.join(parts[2:7]),
            'cmd': ' '.join(parts[7:])
        }
    return result
