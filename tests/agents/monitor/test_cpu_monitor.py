import time
from agents.monitor.cpu_monitor import CPUMonitor


def test_cpu_monitor_collects_status(monitor_server: str):
    monitor = CPUMonitor(server=monitor_server, timeout=5)

    # Accumulate a full window of samples
    window = monitor.cpu.window_size
    for _ in range(window):
        monitor._update_resource()
        time.sleep(0.1)

    cpu_status = monitor.cpu
    assert cpu_status.max_memory is not None
    assert cpu_status.memory_stats is not None
    assert cpu_status.memory_stats['avg'] is not None
    assert cpu_status.cpu_stats is not None
    assert cpu_status.cpu_stats['avg'] is not None
    mem_pct = 100 * cpu_status.memory_stats['avg'] / cpu_status.max_memory
    cpu_util = cpu_status.cpu_stats['avg']
    print(
        "CPU:",
        f"server={cpu_status.server}",
        f"connected={cpu_status.connected}",
        f"memory_pct={mem_pct}",
        f"cpu_avg={cpu_util}",
    )

    assert cpu_status.server == monitor_server
    assert isinstance(cpu_status.connected, bool)

    monitor.stop()
