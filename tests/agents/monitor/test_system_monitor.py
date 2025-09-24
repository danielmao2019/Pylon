import time
import pytest
import torch
from agents.monitor.system_monitor import SystemMonitor


def test_system_monitor_status_snapshot(monitor_server: str, gpu_index: int | None):
    gpu_indices = []
    if gpu_index is not None:
        if monitor_server == 'localhost' and not torch.cuda.is_available():
            pytest.skip("CUDA not available on localhost")
        gpu_indices = [gpu_index]

    monitor = SystemMonitor(server=monitor_server, gpu_indices=gpu_indices, timeout=5)

    # warm up cpu and all gpu monitors
    window = monitor.cpu_monitor.cpu.window_size or 10
    for _ in range(window):
        monitor.cpu_monitor._update_resource()
        for gpu_monitor in monitor.gpu_monitors.values():
            gpu_monitor._update_resource()
        time.sleep(0.1)

    status = monitor.get_system_status()
    cpu_stats = status['cpu']

    assert cpu_stats['max_memory'] is not None
    assert cpu_stats['memory_stats'] is not None
    assert cpu_stats['memory_stats']['avg'] is not None
    assert cpu_stats['cpu_stats'] is not None
    assert cpu_stats['cpu_stats']['avg'] is not None

    cpu_mem_pct = 100 * cpu_stats['memory_stats']['avg'] / cpu_stats['max_memory']
    cpu_util = cpu_stats['cpu_stats']['avg']

    print(
        f"System status for {monitor_server}: cpu_connected={cpu_stats['connected']},"
        f" cpu_memory_pct={cpu_mem_pct}",
        f" cpu_util_avg={cpu_util}",
        f" gpu_count={len(status['gpus'])}"
    )
    for gpu in status['gpus']:
        assert gpu['max_memory'] is not None
        assert gpu.get('memory_stats') is not None
        assert gpu['memory_stats']['avg'] is not None
        assert gpu.get('util_stats') is not None
        assert gpu['util_stats']['avg'] is not None
        gpu_mem_pct = 100 * gpu['memory_stats']['avg'] / gpu['max_memory']
        util_avg = gpu['util_stats']['avg']
        print(
            f"GPU server={gpu['server']} index={gpu['index']} connected={gpu['connected']}"
            f" memory_pct={gpu_mem_pct}"
            f" util_avg={util_avg}"
        )

    assert status['cpu']['server'] == monitor_server
    assert isinstance(status['cpu']['connected'], bool)
    assert isinstance(status['gpus'], list)

    monitor.stop()
