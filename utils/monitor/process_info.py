from typing import List, Dict, TypedDict
import subprocess


class ProcessInfo(TypedDict):
    pid: str
    user: str
    cmd: str
    start_time: str


def get_process_info(server: str, pid: str) -> ProcessInfo:
    """Get process information for a specific PID on a server"""
    cmd = ['ssh', server, 'ps', '-p', pid, '-o', 'pid=,user=,lstart=,cmd=']
    output = subprocess.check_output(cmd).decode().strip()
    parts = output.split()
    return {
        'pid': parts[0],
        'user': parts[1],
        'start_time': ' '.join(parts[2:7]),
        'cmd': ' '.join(parts[7:])
    }


def get_all_processes(server: str) -> Dict[str, ProcessInfo]:
    """Get information for all processes on a server"""
    cmd = ['ssh', server, "ps", "-eo", "pid=,user=,lstart=,cmd="]
    lines = subprocess.check_output(cmd).decode().splitlines()
    result = {}
    for line in lines:
        if "from multiprocessing.spawn import spawn_main; spawn_main" in line:
            continue
        parts = line.strip().split()
        result[parts[0]] = {
            'pid': parts[0],
            'user': parts[1],
            'start_time': ' '.join(parts[2:7]),
            'cmd': ' '.join(parts[7:])
        }
    return result


def get_user_processes(server: str) -> List[str]:
    """Get list of PIDs for processes owned by the server user"""
    cmd = ['ssh', server, 'ps', '-u', server.split('@')[0], '-o', 'pid=']
    return [pid.strip() for pid in subprocess.check_output(cmd).decode().splitlines()]
