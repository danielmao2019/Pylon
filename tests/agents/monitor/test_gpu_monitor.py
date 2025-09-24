import time
import pytest
import torch
from agents.monitor.gpu_monitor import GPUMonitor


@pytest.mark.parametrize("probe_timeout", [5])
def test_gpu_monitor_optional(monitor_server: str, gpu_index: int | None, probe_timeout: int):
    if gpu_index is None:
        pytest.skip("No GPU index provided via --gpu-index")

    if monitor_server == 'localhost' and not torch.cuda.is_available():
        pytest.skip("CUDA not available on localhost")

    monitor = GPUMonitor(server=monitor_server, index=gpu_index, timeout=probe_timeout)

    assert monitor.gpu.window_size is not None
    window = monitor.gpu.window_size
    for _ in range(window):
        monitor._update_resource()
        time.sleep(0.1)

    gpu_status = monitor.gpu
    if not gpu_status.connected:
        pytest.skip("GPU reported as disconnected; skipping detailed assertions")

    assert gpu_status.max_memory is not None
    assert gpu_status.memory_stats is not None
    assert gpu_status.memory_stats['avg'] is not None
    assert gpu_status.util_stats is not None
    assert gpu_status.util_stats['avg'] is not None

    mem_pct = 100 * gpu_status.memory_stats['avg'] / gpu_status.max_memory
    util_avg = gpu_status.util_stats['avg']

    print(
        "GPU status for", monitor_server,
        f"index={gpu_index}",
        f"connected={gpu_status.connected}",
        f"memory_pct={mem_pct}",
        f"util_avg={util_avg}",
    )

    assert gpu_status.server == monitor_server
    assert gpu_status.index == gpu_index

    commands = monitor.get_all_running_commands()
    assert isinstance(commands, list)

    monitor.stop()
