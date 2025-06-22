from typing import List, Dict, TypedDict, Optional, Any
from utils.monitor.process_info import ProcessInfo, get_all_processes
from utils.ssh import SSHConnectionPool
from utils.timeout import with_timeout


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
    connected: bool


def get_server_gpus_mem_util(server: str, gpu_indices: List[int], pool: SSHConnectionPool) -> Dict[int, Dict[str, int]]:
    """Get memory and utilization for multiple GPUs in a single batch operation.

    Args:
        server: The server to query
        indices: List of GPU indices to query
        pool: SSH connection pool instance

    Returns:
        Dict mapping GPU index to dict with 'memory' and 'util' keys
    """
    if not gpu_indices:
        return {}

    gpu_list = ','.join(map(str, gpu_indices))
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
    assert results.keys() == set(gpu_indices), f"{results.keys()=}, {gpu_indices=}"

    return results


def get_server_gpus_processes(server: str, gpu_indices: List[int], pool: SSHConnectionPool) -> Dict[int, List[ProcessInfo]]:
    """Get list of processes running on multiple GPUs in a single batch operation.

    Args:
        server: The server to query
        indices: List of GPU indices to query
        pool: SSH connection pool instance

    Returns:
        Dict mapping GPU index to list of ProcessInfo objects
    """
    if not gpu_indices:
        return {}

    # Get GPU UUIDs for all requested indices
    gpu_list = ','.join(map(str, gpu_indices))
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

    # Get all process information
    all_processes = get_all_processes(server, pool)

    # Map GPU indices to ProcessInfo objects
    results = {}
    for gpu_idx in gpu_indices:
        if gpu_idx in index_to_uuid:
            uuid = index_to_uuid[gpu_idx]
            pids = uuid_to_pids.get(uuid, [])
            processes = [all_processes[pid] for pid in pids if pid in all_processes]
            results[gpu_idx] = processes
        else:
            results[gpu_idx] = []
    assert results.keys() == set(gpu_indices), f"{results.keys()=}, {gpu_indices=}"

    return results


def get_server_gpus_info(server: str, gpu_indices: List[int], pool: SSHConnectionPool, timeout: int = 10) -> Dict[int, Dict[str, Any]]:
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
    assert isinstance(server, str), f"{server=}"
    assert isinstance(gpu_indices, list), f"{gpu_indices=}"
    assert all(isinstance(idx, int) for idx in gpu_indices), f"{gpu_indices=}"
    assert isinstance(pool, SSHConnectionPool), f"{pool=}"
    assert isinstance(timeout, int), f"{timeout=}"

    @with_timeout(seconds=timeout)
    def _get_server_gpus_info():
        results = {}

        # Get memory and utilization for all GPUs in batch
        gpu_mem_util = get_server_gpus_mem_util(server, gpu_indices, pool)

        # Get process info for all GPUs in batch
        gpu_processes = get_server_gpus_processes(server, gpu_indices, pool)

        # Merge query results
        for gpu_idx in gpu_indices:
            # Get processes for this GPU
            results[gpu_idx] = {
                'server': server,
                'index': gpu_idx,
                'max_memory': gpu_mem_util[gpu_idx]['max_memory'],
                'current_memory': gpu_mem_util[gpu_idx]['memory'],
                'current_util': gpu_mem_util[gpu_idx]['util'],
                'processes': gpu_processes[gpu_idx],
                'connected': True,
            }

        return results

    try:
        return _get_server_gpus_info()
    except Exception as e:
        print(f"ERROR: Failed to get GPU info for server {server}, GPUs {gpu_indices}: {e}")
        # Return failed results for all requested GPUs
        return {
            gpu_idx: {
                'server': server,
                'index': gpu_idx,
                'max_memory': None,
                'current_memory': None,
                'current_util': None,
                'processes': None,
                'connected': False,
            }
            for gpu_idx in gpu_indices
        }
