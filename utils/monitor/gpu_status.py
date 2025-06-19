from typing import List, Dict, Any, TypedDict, Optional
from utils.monitor.process_info import ProcessInfo, get_all_processes
from utils.automation.ssh_utils import SSHConnectionPool, SSHCommandError
from utils.timeout import with_timeout


def _build_mapping(output: str) -> Dict[str, List[str]]:
    lines = output.strip().splitlines()
    assert type(lines) == list
    assert all([type(elem) == str for elem in lines])
    lines = [line.strip().split(", ") for line in lines]
    assert all([len(line) == 2 for line in lines])
    result: Dict[str, List[str]] = {}
    for line in lines:
        result[line[0]] = result.get(line[0], []) + [line[1]]
    return result


def get_index2pids(server: str, pool: SSHConnectionPool) -> List[List[str]]:
    index2gpu_uuid_cmd = ['nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader']
    gpu_uuid2pids_cmd = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader']

    index2gpu_uuid_output = pool.execute(server, index2gpu_uuid_cmd)
    gpu_uuid2pids_output = pool.execute(server, gpu_uuid2pids_cmd)

    index2gpu_uuid: Dict[str, List[str]] = _build_mapping(index2gpu_uuid_output)
    gpu_uuid2pids: Dict[str, List[str]] = _build_mapping(gpu_uuid2pids_output)

    num_gpus = len(index2gpu_uuid)
    result: List[List[str]] = [None] * num_gpus
    for idx in range(num_gpus):
        assert len(index2gpu_uuid[str(idx)]) == 1
        gpu_uuid = index2gpu_uuid[str(idx)][0]
        result[idx] = gpu_uuid2pids.get(gpu_uuid, [])
    return result


def get_user_pids(server: str, pool: SSHConnectionPool) -> List[str]:
    cmd = ['ps', '-u', server.split('@')[0], '-o', 'pid=']
    result = pool.execute(server, cmd)
    return [line.strip() for line in result.splitlines()]


def find_running(server: str, pool: SSHConnectionPool, timeout: int = 10) -> List[Dict[str, Any]]:
    r"""This function finds all GPU processes launched by the user.

    Args:
        server: The server to query
        pool: SSH connection pool instance
        timeout: Timeout in seconds for the entire query (default: 10)

    Returns:
        all_running (List[Dict[str, Any]]): a list of dictionaries with the following fields
        {
            server (str): a string in <user_name>@<server_ip> format.
            gpu_index (int): index of GPU on the server.
            command (str): the command this GPU is running by the user.
        }
    """
    @with_timeout(seconds=timeout)
    def _find_running():
        all_running: List[Dict[str, Any]] = []
        gpu_pids = get_index2pids(server, pool)
        user_pids = get_user_pids(server, pool)

        for gpu_index in range(len(gpu_pids)):
            for pid in gpu_pids[gpu_index]:
                if pid not in user_pids:
                    continue
                cmd = ['ps', '-p', pid, '-o', 'cmd=']
                command = pool.execute(server, cmd)
                command_lines = command.splitlines()
                assert len(command_lines) == 1, f"{command_lines=}, {pid=}, {server=}"
                command = command_lines[0].strip()
                if 'python main.py --config-filepath' not in command:
                    continue
                all_running.append({
                    'server': server,
                    'gpu_index': gpu_index,
                    'command': command
                })
        return all_running

    try:
        return _find_running()
    except Exception as e:
        print(f"ERROR: Failed to find running processes for server {server}: {e}")
        return []


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


def get_gpu_mem_util(server: str, indices: List[int], pool: SSHConnectionPool) -> Dict[int, Dict[str, int]]:
    """Get memory and utilization for multiple GPUs in a single batch operation.
    
    Args:
        server: The server to query
        indices: List of GPU indices to query
        pool: SSH connection pool instance
    
    Returns:
        Dict mapping GPU index to dict with 'memory' and 'util' keys
    """
    if not indices:
        return {}
    
    gpu_list = ','.join(map(str, indices))
    cmd = ['nvidia-smi',
           '--query-gpu=index,memory.total,memory.used,utilization.gpu',
           '--format=csv,noheader,nounits',
           f'--id={gpu_list}']
    output = pool.execute(server, cmd)
    
    results = {}
    for line in output.splitlines():
        parts = line.split(', ')
        if len(parts) == 4:
            gpu_idx = int(parts[0])
            max_memory = int(parts[1])
            memory_used = int(parts[2])
            gpu_util = int(parts[3])
            
            results[gpu_idx] = {
                'memory': memory_used,
                'util': gpu_util,
                'max_memory': max_memory
            }
    
    return results


def get_gpu_processes(server: str, indices: List[int], pool: SSHConnectionPool) -> Dict[int, List[str]]:
    """Get list of PIDs running on multiple GPUs in a single batch operation.
    
    Args:
        server: The server to query
        indices: List of GPU indices to query
        pool: SSH connection pool instance
    
    Returns:
        Dict mapping GPU index to list of PIDs
    """
    if not indices:
        return {}
    
    # Get GPU UUIDs for all requested indices
    gpu_list = ','.join(map(str, indices))
    uuid_cmd = ['nvidia-smi', '--query-gpu=index,gpu_uuid',
                '--format=csv,noheader', f'--id={gpu_list}']
    uuid_output = pool.execute(server, uuid_cmd)
    
    # Build mapping from GPU index to UUID
    index_to_uuid = {}
    for line in uuid_output.splitlines():
        parts = line.split(', ')
        if len(parts) == 2:
            gpu_idx = int(parts[0])
            gpu_uuid = parts[1]
            index_to_uuid[gpu_idx] = gpu_uuid
    
    # Get all compute apps (PIDs) for all GPUs
    pids_cmd = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader']
    pids_output = pool.execute(server, pids_cmd)
    
    # Build mapping from UUID to PIDs
    uuid_to_pids = {}
    for line in pids_output.splitlines():
        parts = line.split(', ')
        if len(parts) == 2:
            uuid, pid = parts[0], parts[1]
            if uuid not in uuid_to_pids:
                uuid_to_pids[uuid] = []
            uuid_to_pids[uuid].append(pid)
    
    # Map GPU indices to PIDs
    results = {}
    for gpu_idx in indices:
        if gpu_idx in index_to_uuid:
            uuid = index_to_uuid[gpu_idx]
            results[gpu_idx] = uuid_to_pids.get(uuid, [])
        else:
            results[gpu_idx] = []
    
    return results


def get_all_gpu_info_batched(server: str, gpu_indices: List[int], pool: SSHConnectionPool, timeout: int = 10) -> Dict[int, Dict]:
    """Get information for multiple GPUs on a server in a single batch operation.
    
    This reduces SSH connection overhead by batching multiple GPU queries.
    
    Args:
        server: The server to query
        gpu_indices: List of GPU indices to query
        pool: SSH connection pool instance
        timeout: Timeout in seconds for the entire batch query (default: 10)
    
    Returns:
        Dict mapping GPU index to GPU info dictionary
    """
    @with_timeout(seconds=timeout)
    def _get_all_gpu_info_batched():
        results = {}
        
        # Get memory and utilization for all GPUs in batch
        gpu_mem_util = get_gpu_mem_util(server, gpu_indices, pool)
        
        # Get process info for all GPUs in batch
        gpu_pids = get_gpu_processes(server, gpu_indices, pool)
        all_processes = get_all_processes(server, pool)
        
        # Process each GPU
        for gpu_idx in gpu_indices:
            if gpu_idx in gpu_mem_util:
                mem_util_data = gpu_mem_util[gpu_idx]
                
                # Get processes for this GPU
                pids = gpu_pids.get(gpu_idx, [])
                processes = [all_processes[pid] for pid in pids if pid in all_processes]
                
                results[gpu_idx] = {
                    'server': server,
                    'index': gpu_idx,
                    'max_memory': mem_util_data['max_memory'],
                    'processes': processes,
                    'current_memory': mem_util_data['memory'],
                    'current_util': mem_util_data['util'],
                    'success': True,
                }
            else:
                # Handle case where GPU data wasn't returned
                results[gpu_idx] = {
                    'server': server,
                    'index': gpu_idx,
                    'max_memory': None,
                    'processes': None,
                    'current_memory': None,
                    'current_util': None,
                    'success': False,
                }
        
        return results

    try:
        return _get_all_gpu_info_batched()
    except Exception as e:
        print(f"ERROR: Failed to get batch GPU info for server {server}, GPUs {gpu_indices}: {e}")
        # Return failed results for all requested GPUs
        return {
            gpu_idx: {
                'server': server,
                'index': gpu_idx,
                'max_memory': None,
                'processes': None,
                'current_memory': None,
                'current_util': None,
                'success': False,
            }
            for gpu_idx in gpu_indices
        }


def get_ssh_pool_status(pool: SSHConnectionPool) -> Dict[str, Dict[str, int]]:
    """Get SSH connection pool status for monitoring purposes."""
    return pool.get_stats()
