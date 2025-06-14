from typing import List, Dict, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
from utils.monitor.gpu_status import GPUStatus, get_gpu_info


class GPUMonitor:

    def __init__(self, gpus: List[GPUStatus], timeout: int = 5):
        self.gpus = gpus
        self.monitor_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=len(gpus))
        self.timeout = timeout

        # Do one update first
        list(self.executor.map(self._update_gpu_info, self.gpus))

    def start(self):
        """Starts background monitoring thread that continuously updates GPU info"""
        def monitor_loop():
            while True:
                list(self.executor.map(self._update_gpu_info, self.gpus))

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _update_gpu_info(self, gpu):
        """Updates information for a single GPU"""
        # Get current GPU info
        current_info = get_gpu_info(gpu['server'], gpu['index'], timeout=self.timeout)

        # Update connection status
        gpu['connected'] = current_info['success']

        # If query failed, set all fields to None except server and index
        if not current_info['success']:
            gpu['max_memory'] = None
            gpu['processes'] = None
            gpu['memory_window'] = None
            gpu['util_window'] = None
            gpu['memory_stats'] = None
            gpu['util_stats'] = None
            return

        # Update processes
        gpu['processes'] = current_info['processes']

        # Update max memory
        gpu['max_memory'] = current_info['max_memory']

        # Initialize windows if they are None (GPU just reconnected)
        if gpu['memory_window'] is None:
            gpu['memory_window'] = []
        if gpu['util_window'] is None:
            gpu['util_window'] = []

        # Update rolling windows
        gpu['memory_window'].append(current_info['current_memory'])
        gpu['util_window'].append(current_info['current_util'])

        # Trim windows if needed
        if len(gpu['memory_window']) > gpu['window_size']:
            gpu['memory_window'] = gpu['memory_window'][-gpu['window_size']:]
        if len(gpu['util_window']) > gpu['window_size']:
            gpu['util_window'] = gpu['util_window'][-gpu['window_size']:]

        # Update stats
        gpu['memory_stats'] = {
            'min': min(gpu['memory_window']),
            'max': max(gpu['memory_window']),
            'avg': sum(gpu['memory_window']) / len(gpu['memory_window']),
        }
        gpu['util_stats'] = {
            'min': min(gpu['util_window']),
            'max': max(gpu['util_window']),
            'avg': sum(gpu['util_window']) / len(gpu['util_window']),
        }

    def _check(self) -> Dict:
        """Returns current status of all GPUs without rolling windows"""
        return {
            f"{gpu['server']}:{gpu['index']}": {
                'server': gpu['server'],
                'index': gpu['index'],
                'max_memory': gpu['max_memory'],
                'memory_min': gpu['memory_stats']['min'] if gpu['memory_stats'] is not None else None,
                'memory_max': gpu['memory_stats']['max'] if gpu['memory_stats'] is not None else None,
                'memory_avg': gpu['memory_stats']['avg'] if gpu['memory_stats'] is not None else None,
                'util_min': gpu['util_stats']['min'] if gpu['util_stats'] is not None else None,
                'util_max': gpu['util_stats']['max'] if gpu['util_stats'] is not None else None,
                'util_avg': gpu['util_stats']['avg'] if gpu['util_stats'] is not None else None,
                'connected': gpu['connected'],
            }
            for gpu in self.gpus
        }

    def log_stats(self, logger):
        """Logs status of all monitored GPUs"""
        stats = self._check()
        assert len(stats) == 1, "Only support single GPU training for now."
        stats = list(stats.values())[0]
        logger.update_buffer(stats)
