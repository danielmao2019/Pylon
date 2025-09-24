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
    status = monitor.get_system_status()

    print(
        f"System status for {monitor_server}: cpu_connected={status['cpu']['connected']},"
        f" gpu_count={len(status['gpus'])}"
    )

    assert status['cpu']['server'] == monitor_server
    assert isinstance(status['cpu']['connected'], bool)
    assert isinstance(status['gpus'], list)

    monitor.stop()
