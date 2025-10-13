import time
import pytest
from agents.monitor.gpu_monitor import GPUMonitor


@pytest.mark.parametrize("probe_timeout", [5])
def test_gpu_monitor_optional(
    monitor_server: str, gpu_index: int | None, probe_timeout: int
):
    if gpu_index is None:
        gpu_index = 0

    monitor = GPUMonitor(server=monitor_server, index=gpu_index, timeout=probe_timeout)

    try:
        window = monitor.window_size
        for _ in range(window):
            monitor._update_resource()
            time.sleep(0.1)

        gpu_status = monitor.gpu
        assert gpu_status.connected, "GPU reported as disconnected; monitor check failed"

        assert gpu_status.window_size == monitor.window_size
        assert gpu_status.max_memory is not None
        assert gpu_status.memory_stats is not None
        assert gpu_status.memory_stats['avg'] is not None
        assert gpu_status.util_stats is not None
        assert gpu_status.util_stats['avg'] is not None

        mem_pct = 100 * gpu_status.memory_stats['avg'] / gpu_status.max_memory
        util_avg = gpu_status.util_stats['avg']

        print(
            "GPU status for",
            monitor_server,
            f"index={gpu_index}",
            f"connected={gpu_status.connected}",
            f"memory_pct={mem_pct}",
            f"util_avg={util_avg}",
        )

        assert gpu_status.server == monitor_server
        assert gpu_status.index == gpu_index

        commands = monitor.get_all_running_commands()
        assert isinstance(commands, list)
    finally:
        monitor.stop()
