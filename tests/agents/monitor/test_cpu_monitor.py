import pytest
from agents.monitor.cpu_monitor import CPUMonitor


def test_cpu_monitor_collects_status(monitor_server: str):
    monitor = CPUMonitor(server=monitor_server, timeout=5)
    cpu_status = monitor.cpu

    print(f"CPU status for {monitor_server}: connected={cpu_status.connected}, cores={cpu_status.cpu_cores}")

    assert cpu_status.server == monitor_server
    assert isinstance(cpu_status.connected, bool)

    # Basic sanity check on commands helper
    commands = monitor.get_all_running_commands()
    assert isinstance(commands, list)

    monitor.stop()
