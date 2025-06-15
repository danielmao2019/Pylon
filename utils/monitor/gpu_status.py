from typing import List, Dict, Any, TypedDict, Optional
import subprocess
from utils.monitor.process_info import ProcessInfo, get_all_processes


def _build_mapping(output: str) -> Dict[str, List[str]]:
    lines = output.decode().strip().splitlines()
    assert type(lines) == list
    assert all([type(elem) == str for elem in lines])
    lines = [line.strip().split(", ") for line in lines]
    assert all([len(line) == 2 for line in lines])
    result: Dict[str, List[str]] = {}
    for line in lines:
        result[line[0]] = result.get(line[0], []) + [line[1]]
    return result


def get_index2pids(server: str) -> List[List[str]]:
    index2gpu_uuid_cmd = ['nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader']
    gpu_uuid2pids_cmd = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader']
    if server != 'localhost':
        index2gpu_uuid_cmd = ['ssh', server] + index2gpu_uuid_cmd
        gpu_uuid2pids_cmd = ['ssh', server] + gpu_uuid2pids_cmd
    index2gpu_uuid: Dict[str, List[str]] = _build_mapping(subprocess.check_output(index2gpu_uuid_cmd))
    gpu_uuid2pids: Dict[str, List[str]] = _build_mapping(subprocess.check_output(gpu_uuid2pids_cmd))
    num_gpus = len(index2gpu_uuid)
    result: List[List[str]] = [None] * num_gpus
    for idx in range(num_gpus):
        assert len(index2gpu_uuid[str(idx)]) == 1
        gpu_uuid = index2gpu_uuid[str(idx)][0]
        result[idx] = gpu_uuid2pids.get(gpu_uuid, [])
    return result


def get_user_pids(server: str) -> List[str]:
    cmd = ['ps', '-u', server.split('@')[0], '-o', 'pid=']
    if server != 'localhost':
        cmd = ['ssh', server] + cmd
    result: List[str] = subprocess.check_output(cmd).decode().splitlines()
    result = list(map(lambda x: x.strip(), result))
    return result


def find_running(server: str) -> List[Dict[str, Any]]:
    r"""This function finds all GPU processes launched by the user.

    Returns:
        all_running (List[Dict[str, Any]]): a list of dictionaries with the following fields
        {
            server (str): a string in <user_name>@<server_ip> format.
            gpu_index (int): index of GPU on the server.
            command (str): the command this GPU is running by the user.
        }
    """
    all_running: List[Dict[str, Any]] = []
    gpu_pids = get_index2pids(server)
    user_pids = get_user_pids(server)
    for gpu_index in range(len(gpu_pids)):
        for pid in gpu_pids[gpu_index]:
            if pid not in user_pids:
                continue
            cmd = ['ps', '-p', pid, '-o', 'cmd=']
            if server != 'localhost':
                cmd = ['ssh', server] + cmd
            try:
                command = subprocess.check_output(cmd).decode().splitlines()
            except:
                continue
            assert len(command) == 1, f"{command=}, {pid=}, {server=}"
            command = command[0].strip()
            if 'python main.py --config-filepath' not in command:
                continue
            all_running.append({
                'server': server,
                'gpu_index': gpu_index,
                'command': command
            })
    return all_running


class GPUStatus(TypedDict):
    server: str
    index: int
    max_memory: int
    processes: List[ProcessInfo]
    window_size: int
    memory_window: List[int]
    util_window: List[int]
    memory_stats: dict[str, Optional[float]]
    util_stats: dict[str, Optional[float]]


def get_gpu_memory(server: str, gpu_index: int, timeout: int = 5) -> int:
    """Get total memory for a specific GPU"""
    cmd = ['nvidia-smi',
           '--query-gpu=memory.total',
           '--format=csv,noheader,nounits',
           f'--id={gpu_index}']
    if server != 'localhost':
        cmd = ['ssh', server] + cmd
    output = subprocess.check_output(cmd, timeout=timeout).decode().strip()
    return int(output)


def get_gpu_utilization(server: str, gpu_index: int, timeout: int = 5) -> Dict[str, int]:
    """Get current memory and utilization for a specific GPU"""
    cmd = ['nvidia-smi',
           '--query-gpu=memory.used,utilization.gpu',
           '--format=csv,noheader,nounits',
           f'--id={gpu_index}']
    if server != 'localhost':
        cmd = ['ssh', server] + cmd
    output = subprocess.check_output(cmd, timeout=timeout).decode().strip()
    memory_used, gpu_util = map(int, output.split(', '))
    return {'memory': memory_used, 'util': gpu_util}


def get_gpu_processes(server: str, gpu_index: int, timeout: int = 5) -> List[str]:
    """Get list of PIDs running on a specific GPU"""
    # Get GPU UUID
    uuid_cmd = ['nvidia-smi', '--query-gpu=index,gpu_uuid',
                '--format=csv,noheader', f'--id={gpu_index}']
    if server != 'localhost':
        uuid_cmd = ['ssh', server] + uuid_cmd
    uuid_output = subprocess.check_output(uuid_cmd, timeout=timeout).decode().strip()
    gpu_uuid = uuid_output.split(', ')[1]

    # Get PIDs for this UUID
    pids_cmd = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader']
    if server != 'localhost':
        pids_cmd = ['ssh', server] + pids_cmd
    pids_output = subprocess.check_output(pids_cmd, timeout=timeout).decode().strip()
    pids = []
    for line in pids_output.splitlines():
        uuid, pid = line.split(', ')
        if uuid == gpu_uuid:
            pids.append(pid)
    return pids


def get_gpu_info(server: str, gpu_index: int, timeout: int = 5) -> Dict:
    """Get all information for a specific GPU

    Args:
        server: The server to query
        gpu_index: The GPU index to query
        timeout: Timeout in seconds for GPU queries (default: 5)

    Returns:
        Dict containing GPU info with an additional 'success' field indicating if the query succeeded
    """
    try:
        # Get basic GPU info
        max_memory = get_gpu_memory(server, gpu_index, timeout)
        util_info = get_gpu_utilization(server, gpu_index, timeout)

        # Get process info
        gpu_pids = get_gpu_processes(server, gpu_index, timeout)
        all_processes = get_all_processes(server)
        processes = [all_processes[pid] for pid in gpu_pids if pid in all_processes]

        return {
            'server': server,
            'index': gpu_index,
            'max_memory': max_memory,
            'processes': processes,
            'current_memory': util_info['memory'],
            'current_util': util_info['util'],
            'success': True,
        }
    except subprocess.TimeoutExpired:
        return {
            'server': server,
            'index': gpu_index,
            'max_memory': None,
            'processes': None,
            'current_memory': None,
            'current_util': None,
            'success': False,
        }
    except Exception:
        return {
            'server': server,
            'index': gpu_index,
            'max_memory': None,
            'processes': None,
            'current_memory': None,
            'current_util': None,
            'success': False,
        }
