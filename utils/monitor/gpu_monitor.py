import subprocess
import time
import torch
import os


def monitor_gpu_usage():
    """
    Monitor GPU usage for the specified device using both nvidia-smi and PyTorch.

    Returns:
        dict: Dictionary containing GPU usage information including:
            - index: GPU index (as seen by PyTorch)
            - physical_index: Physical GPU index (actual GPU)
            - name: GPU architecture name
            - memory_allocated: Memory allocated by PyTorch in MB
            - memory_reserved: Memory reserved by PyTorch in MB
            - memory_used: Memory used according to nvidia-smi in MB
            - memory_total: Total memory in MB
            - memory_util: Memory utilization percentage
            - gpu_util: GPU utilization percentage
    """
    # Get the current device index
    device_index = torch.cuda.current_device()

    # Get the actual physical GPU index from CUDA_VISIBLE_DEVICES
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices:
        # Parse the CUDA_VISIBLE_DEVICES string
        visible_devices = [int(d.strip()) for d in cuda_visible_devices.split(',')]
        # Map the logical device index to the physical device index
        physical_device_index = visible_devices[device_index]
    else:
        physical_device_index = device_index

    # Get GPU name
    gpu_name = torch.cuda.get_device_name(device_index)

    # Get PyTorch memory stats
    memory_allocated = torch.cuda.memory_allocated(device_index) / (1024 * 1024)  # Convert to MB
    memory_used_pytorch = torch.cuda.memory_reserved(device_index) / (1024 * 1024)  # Convert to MB
    memory_cache = memory_used_pytorch - memory_allocated
    memory_total_pytorch = torch.cuda.get_device_properties(device_index).total_memory / (1024 * 1024)  # Convert to MB

    # Get nvidia-smi stats
    util_cmd = ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits', f'--id={physical_device_index}']
    util_output = subprocess.check_output(util_cmd).decode().strip()
    gpu_util, memory_used_nvidia, memory_total_nvidia = map(int, util_output.split(', '))

    # Assert that memory used from nvidia-smi is close to memory reserved from PyTorch
    assert abs(memory_used_nvidia - memory_used_pytorch) < 1, \
        f"Memory used/reserved mismatch: nvidia-smi={memory_used_nvidia}, PyTorch={memory_used_pytorch}"

    # Assert that memory total is the same from both sources
    assert abs(memory_total_pytorch - memory_total_nvidia) < 1, \
        f"Memory total mismatch: PyTorch={memory_total_pytorch}, nvidia-smi={memory_total_nvidia}"

    # Initialize result with all information
    result = {
        'index': device_index,  # Logical index (as seen by PyTorch)
        'physical_index': physical_device_index,  # Physical index (actual GPU)
        'name': gpu_name,
        'memory_allocated': memory_allocated,
        'memory_cache': memory_cache,
        'memory_used': memory_used_nvidia,
        'memory_total': memory_total_nvidia,
        'memory_util': (memory_used_nvidia / memory_total_nvidia) * 100 if memory_total_nvidia > 0 else 0,
        'gpu_util': gpu_util
    }

    return result


class GPUMonitor:
    """
    Class to track GPU usage over time, recording min/max values.
    """
    def __init__(self):
        """Initialize the GPU monitor."""
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
        stats = monitor_gpu_usage()
        self.samples.append(stats)

        # Update min/max memory
        if stats['memory_used'] < self.min_memory:
            self.min_memory = stats['memory_used']
        if stats['memory_used'] > self.max_memory:
            self.max_memory = stats['memory_used']

        # Update min/max utilization
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
        """
        Log current GPU stats to the provided logger.

        Args:
            logger: The logger to log to (can be traditional logger or screen logger)
        """
        # Update the logger with GPU stats
        stats = self.get_stats()

        # Update the buffer with GPU stats for both traditional and screen loggers
        logger.update_buffer(stats)
