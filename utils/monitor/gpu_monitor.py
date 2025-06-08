from typing import List, Dict, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
from utils.monitor.gpu_status import GPUStatus, get_gpu_info


class GPUMonitor:

    def __init__(self, gpus: List[GPUStatus]):
        self.gpus = gpus
        self.monitor_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=len(gpus))

    def start(self):
        """Starts background monitoring thread that continuously updates GPU info"""
        def monitor_loop():
            while True:
                # Create a list to store futures
                futures = []
                
                # Submit monitoring tasks for each GPU
                for gpu in self.gpus:
                    future = self.executor.submit(self._update_gpu_info, gpu)
                    futures.append(future)
                
                # Wait for all futures to complete
                for future in futures:
                    future.result()

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _update_gpu_info(self, gpu):
        """Updates information for a single GPU"""
        # Get current GPU info
        current_info = get_gpu_info(gpu['server'], gpu['index'])

        # Update processes
        gpu['processes'] = current_info['processes']

        # Update max memory
        gpu['max_memory'] = current_info['max_memory']

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
            'avg': sum(gpu['memory_window']) / len(gpu['memory_window'])
        }
        gpu['util_stats'] = {
            'min': min(gpu['util_window']),
            'max': max(gpu['util_window']),
            'avg': sum(gpu['util_window']) / len(gpu['util_window'])
        }

    def check(self) -> Dict:
        """Returns current status of all GPUs without rolling windows"""
        return {
            f"{gpu['server']}:{gpu['index']}": {
                'server': gpu['server'],
                'index': gpu['index'],
                'max_memory': gpu['max_memory'],
                'memory_min': gpu['memory_stats']['min'],
                'memory_max': gpu['memory_stats']['max'],
                'memory_avg': gpu['memory_stats']['avg'],
                'util_min': gpu['util_stats']['min'],
                'util_max': gpu['util_stats']['max'],
                'util_avg': gpu['util_stats']['avg']
            }
            for gpu in self.gpus
        }

    def log_stats(self, logger):
        """Logs status of all monitored GPUs"""
        stats = self.check()
        logger.update_buffer(stats)
