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


def _get_index2util(server: str) -> List[Dict[str, int]]:
    fmem_cmd = ['ssh', server, 'nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits']
    util_cmd = ['ssh', server, 'nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits']
    fmem_out: Dict[str, List[str]] = _build_mapping(subprocess.check_output(fmem_cmd))
    util_out: Dict[str, List[str]] = _build_mapping(subprocess.check_output(util_cmd))
    index2fmem: List[int] = []
    index2util: List[int] = []
    for index in fmem_out:
        assert len(fmem_out[index]) == 1
        index2fmem.append(int(fmem_out[index][0]))
    for index in util_out:
        assert len(util_out[index]) == 1
        index2util.append(int(util_out[index][0]))
    result: List[Dict[str, int]] = [{
        'fmem': fmem,
        'util': util,
    } for fmem, util in zip(index2fmem, index2util)]
    return result


def get_index2pids(server: str) -> List[List[str]]:
    index2gpu_uuid_cmd = ['ssh', server, 'nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader']
    gpu_uuid2pids_cmd = ['ssh', server, 'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader']
    index2gpu_uuid: Dict[str, List[str]] = _build_mapping(subprocess.check_output(index2gpu_uuid_cmd))
    gpu_uuid2pids: Dict[str, List[str]] = _build_mapping(subprocess.check_output(gpu_uuid2pids_cmd))
    num_gpus = len(index2gpu_uuid)
    result: List[List[str]] = [None] * num_gpus
    for idx in range(num_gpus):
        assert len(index2gpu_uuid[str(idx)]) == 1
        gpu_uuid = index2gpu_uuid[str(idx)][0]
        result[idx] = gpu_uuid2pids.get(gpu_uuid, [])
    return result


def get_all_p(server: str) -> Dict[str, Dict[str, str]]:
    cmd = ['ssh', server, "ps", "-eo", "pid=,user=,lstart=,cmd="]
    lines = subprocess.check_output(cmd).decode().splitlines()
    result: Dict[str, Dict[str, str]] = {}
    for line in lines:
        if "from multiprocessing.spawn import spawn_main; spawn_main" in line:
            continue
        parts = line.strip().split()
        result[parts[0]] = {'pid': parts[0], 'user': parts[1], 'start': ' '.join(parts[2:7]), 'cmd': ' '.join(parts[7:])}
    return result


def get_server_status(server: str) -> List[Dict[str, Any]]:
    index2pids = get_index2pids(server)
    all_p = get_all_p(server)
    index2util = _get_index2util(server)
    result: List[Dict[str, Any]] = [{
        'processes': [all_p[pid] for pid in pids if pid in all_p],
        'util': util,
    } for pids, util in zip(index2pids, index2util)]
    return result


def get_user_pids(server: str) -> List[str]:
    cmd = ['ssh', server, 'ps', '-u', server.split('@')[0], '-o', 'pid=']
    result: List[str] = subprocess.check_output(cmd).decode().splitlines()
    return list(map(lambda x: x.strip(), result))


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
            cmd = ['ssh', server, 'ps', '-p', pid, '-o', 'cmd=']
            try:
                command = subprocess.check_output(cmd).decode().splitlines()
            except:
                continue
            assert len(command) == 1, f"{command=}, {pid=}"
            command = command[0].strip()
            if "from multiprocessing.spawn import spawn_main; spawn_main" in command:
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


def get_gpu_memory(server: str, gpu_index: int) -> int:
    """Get total memory for a specific GPU"""
    cmd = ['ssh', server, 'nvidia-smi',
           '--query-gpu=memory.total',
           '--format=csv,noheader,nounits',
           f'--id={gpu_index}']
    output = subprocess.check_output(cmd).decode().strip()
    return int(output)


def get_gpu_utilization(server: str, gpu_index: int) -> Dict[str, int]:
    """Get current memory and utilization for a specific GPU"""
    cmd = ['ssh', server, 'nvidia-smi',
           '--query-gpu=memory.used,utilization.gpu',
           '--format=csv,noheader,nounits',
           f'--id={gpu_index}']
    output = subprocess.check_output(cmd).decode().strip()
    memory_used, gpu_util = map(int, output.split(', '))
    return {'memory': memory_used, 'util': gpu_util}


def get_gpu_processes(server: str, gpu_index: int) -> List[str]:
    """Get list of PIDs running on a specific GPU"""
    # Get GPU UUID
    uuid_cmd = ['ssh', server, 'nvidia-smi', '--query-gpu=index,gpu_uuid',
                '--format=csv,noheader', f'--id={gpu_index}']
    uuid_output = subprocess.check_output(uuid_cmd).decode().strip()
    gpu_uuid = uuid_output.split(', ')[1]

    # Get PIDs for this UUID
    pids_cmd = ['ssh', server, 'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader']
    pids_output = subprocess.check_output(pids_cmd).decode().strip()
    pids = []
    for line in pids_output.splitlines():
        uuid, pid = line.split(', ')
        if uuid == gpu_uuid:
            pids.append(pid)
    return pids


def get_gpu_info(server: str, gpu_index: int) -> Dict:
    """Get all information for a specific GPU"""
    # Get basic GPU info
    max_memory = get_gpu_memory(server, gpu_index)
    util_info = get_gpu_utilization(server, gpu_index)

    # Get process info
    gpu_pids = get_gpu_processes(server, gpu_index)
    all_processes = get_all_processes(server)
    processes = [all_processes[pid] for pid in gpu_pids if pid in all_processes]

    return {
        'server': server,
        'index': gpu_index,
        'max_memory': max_memory,
        'processes': processes,
        'current_memory': util_info['memory'],
        'current_util': util_info['util']
    }
