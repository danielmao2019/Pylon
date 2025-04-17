import subprocess
import time
import torch
import os


def monitor_gpu_usage(device_index=None):
    """
    Monitor GPU usage for the specified device or all devices.

    Args:
        device_index (int, optional): Specific GPU index to monitor. If None, monitors all GPUs.

    Returns:
        dict: Dictionary containing GPU usage information including:
            - index: GPU index
            - name: GPU architecture name
            - memory_used: Current memory usage in MB
            - memory_total: Total memory in MB
            - memory_util: Memory utilization percentage
            - gpu_util: GPU utilization percentage
    """
    # Get current device if not specified
    if device_index is None:
        device_index = torch.cuda.current_device()

    # Get GPU name
    gpu_name = torch.cuda.get_device_name(device_index)

    # Get memory info
    memory_allocated = torch.cuda.memory_allocated(device_index) / (1024 * 1024)  # Convert to MB
    memory_reserved = torch.cuda.memory_reserved(device_index) / (1024 * 1024)  # Convert to MB
    memory_total = torch.cuda.get_device_properties(device_index).total_memory / (1024 * 1024)  # Convert to MB

    # Get the actual physical GPU index from CUDA_VISIBLE_DEVICES
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices:
        # Parse the CUDA_VISIBLE_DEVICES string
        visible_devices = [int(d.strip()) for d in cuda_visible_devices.split(',')]
        # Map the logical device index to the physical device index
        physical_device_index = visible_devices[device_index]
    else:
        physical_device_index = device_index

    # Prepare the common dictionary with PyTorch metrics
    result = {
        'index': device_index,  # Logical index (as seen by PyTorch)
        'physical_index': physical_device_index,  # Physical index (actual GPU)
        'name': gpu_name,
        'memory_used': memory_allocated,
        'memory_total': memory_total,
        'memory_util': (memory_allocated / memory_total) * 100 if memory_total > 0 else 0,
        'gpu_util': None,  # Will be updated if nvidia-smi succeeds
        'torch_memory_allocated': memory_allocated,
        'torch_memory_reserved': memory_reserved
    }

    # Try to get additional metrics from nvidia-smi
    try:
        # Query GPU utilization using the physical device index
        util_cmd = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits', f'--id={physical_device_index}']
        util_output = subprocess.check_output(util_cmd).decode().strip()
        gpu_util, mem_used, mem_total = map(int, util_output.split(', ')[1:])

        # Update the result with nvidia-smi data
        result.update({
            'memory_used': mem_used,
            'memory_total': mem_total,
            'memory_util': (mem_used / mem_total) * 100 if mem_total > 0 else 0,
            'gpu_util': gpu_util
        })
    except Exception:
        # If nvidia-smi fails, we already have the PyTorch metrics in the result
        pass

    return result


class GPUMonitor:
    """
    Class to track GPU usage over time, recording min/max values.
    """
    def __init__(self, device_index=None):
        self.device_index = device_index
        self.start_time = None
        self.min_memory = float('inf')
        self.max_memory = 0
        self.min_util = float('inf')
        self.max_util = 0
        self.samples = []

    def start(self):
        """Start monitoring GPU usage."""
        self.start_time = time.time()
        self.min_memory = float('inf')
        self.max_memory = 0
        self.min_util = float('inf')
        self.max_util = 0
        self.samples = []
        self._update()

    def _update(self):
        """Update current GPU stats."""
        stats = monitor_gpu_usage(self.device_index)
        self.samples.append(stats)

        # Update min/max memory
        if stats['memory_used'] < self.min_memory:
            self.min_memory = stats['memory_used']
        if stats['memory_used'] > self.max_memory:
            self.max_memory = stats['memory_used']

        # Update min/max utilization if available
        if stats['gpu_util'] is not None:
            if stats['gpu_util'] < self.min_util:
                self.min_util = stats['gpu_util']
            if stats['gpu_util'] > self.max_util:
                self.max_util = stats['gpu_util']

    def update(self):
        """Update current GPU stats."""
        self._update()

    def get_stats(self):
        """Get current GPU stats with min/max values."""
        self._update()  # Get latest stats
        current = self.samples[-1]

        return {
            'index': current['index'],
            'physical_index': current['physical_index'],
            'name': current['name'],
            'current_memory': current['memory_used'],
            'min_memory': self.min_memory,
            'max_memory': self.max_memory,
            'current_util': current['gpu_util'],
            'min_util': self.min_util if self.min_util != float('inf') else None,
            'max_util': self.max_util,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }

    def log_stats(self, logger):
        """Log current GPU stats to the provided logger."""
        stats = self.get_stats()

        logger.update_buffer({
            f"gpu_{stats['index']}_name": stats['name'],
            f"gpu_{stats['index']}_physical_index": stats['physical_index'],
            f"gpu_{stats['index']}_current_memory_mb": round(stats['current_memory'], 2),
            f"gpu_{stats['index']}_min_memory_mb": round(stats['min_memory'], 2),
            f"gpu_{stats['index']}_max_memory_mb": round(stats['max_memory'], 2),
        })

        if stats['current_util'] is not None:
            logger.update_buffer({
                f"gpu_{stats['index']}_current_util_percent": round(stats['current_util'], 2),
                f"gpu_{stats['index']}_min_util_percent": round(stats['min_util'], 2) if stats['min_util'] is not None else None,
                f"gpu_{stats['index']}_max_util_percent": round(stats['max_util'], 2),
            })
