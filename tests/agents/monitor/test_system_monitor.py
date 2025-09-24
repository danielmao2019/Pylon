import time
from agents.monitor.system_monitor import SystemMonitor


def test_system_monitor_status_snapshot(monitor_server: str, gpu_index: int | None):
    monitor = SystemMonitor(server=monitor_server, timeout=5)

    # warm up cpu and all gpu monitors
    window = monitor.window_size
    for _ in range(window):
        monitor.cpu_monitor._update_resource()
        for gpu_monitor in monitor.gpu_monitors.values():
            gpu_monitor._update_resource()
        time.sleep(0.1)

    status = monitor.get_system_status()
    cpu_stats = status['cpu']

    if monitor.cpu_monitor.cpu.connected:
        assert monitor.cpu_monitor.cpu.window_size == monitor.window_size
    else:
        assert monitor.cpu_monitor.cpu.window_size is None

    if cpu_stats['connected']:
        assert cpu_stats['window_size'] == monitor.window_size
    else:
        assert cpu_stats['window_size'] is None
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
    gpu_statuses = status['gpus']
    if gpu_index is not None:
        assert any(gpu['index'] == gpu_index for gpu in gpu_statuses)

    for idx, gpu in enumerate(gpu_statuses):
        gpu_monitor = monitor.gpu_monitors.get(gpu['index'])
        if gpu_monitor is not None:
            if gpu_monitor.gpu.connected:
                assert gpu_monitor.gpu.window_size == monitor.window_size
            else:
                assert gpu_monitor.gpu.window_size is None

        if gpu['connected']:
            assert gpu['window_size'] == monitor.window_size
        else:
            assert gpu['window_size'] is None
        assert gpu['max_memory'] is not None
        assert gpu['memory_stats'] is not None
        assert gpu['memory_stats']['avg'] is not None
        assert gpu['util_stats'] is not None
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
