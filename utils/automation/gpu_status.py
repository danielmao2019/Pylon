from typing import List, Dict, Any
import subprocess


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
    cmd = ['ssh', server, "ps", "-eo", "pid,user,lstart,cmd"]
    out = subprocess.check_output(cmd)
    lines = out.decode().strip().splitlines()[1:]
    result: Dict[str, Dict[str, str]] = {}
    for line in lines:
        if "from multiprocessing.spawn import spawn_main; spawn_main" in line:
            continue
        parts = line.split()
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
    cmd = ['ssh', server, 'ps', '-u', server.split('@')[0], '-eo', 'pid']
    outputs = subprocess.check_output(cmd).decode().strip()
    result: List[str] = list(map(lambda x: x.strip(), outputs.split('\n')))
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
            cmd = ['ssh', server, 'ps', '-p', pid, '-eo', 'cmd']
            running = subprocess.check_output(cmd).decode().strip()
            all_running.append({
                'server': server,
                'gpu_index': gpu_index,
                'command': running
            })
    return all_running
