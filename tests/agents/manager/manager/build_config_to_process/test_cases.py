import pytest

from agents.manager.manager import Manager
from agents.monitor.cpu_status import CPUStatus
from agents.monitor.gpu_status import GPUStatus
from agents.monitor.process_info import ProcessInfo


def test_manager_build_command_to_process_mapping():
    """Test building command to ProcessInfo mapping from GPU data."""
    connected_gpus = [
        GPUStatus(
            server='test_server',
            index=0,
            window_size=10,
            max_memory=0,
            processes=[
                ProcessInfo(
                    pid='12345',
                    user='testuser',
                    cmd='python main.py --config-filepath configs/exp1.py',
                    start_time='Mon Jan  1 10:00:00 2024',
                ),
                ProcessInfo(
                    pid='12346',
                    user='testuser',
                    cmd='some_other_process --not-config',
                    start_time='Mon Jan  1 11:00:00 2024',
                ),
            ],
        ),
        GPUStatus(
            server='test_server',
            index=1,
            window_size=10,
            max_memory=0,
            processes=[
                ProcessInfo(
                    pid='54321',
                    user='testuser',
                    cmd='python main.py --config-filepath configs/exp2.py --debug',
                    start_time='Mon Jan  1 12:00:00 2024',
                )
            ],
        ),
    ]

    mapping = Manager.build_command_to_process_mapping(connected_gpus)

    # Should include actual command strings observed on the GPUs
    assert len(mapping) == 3
    exp1_command = "python main.py --config-filepath configs/exp1.py"
    exp2_command = "python main.py --config-filepath configs/exp2.py --debug"
    other_command = "some_other_process --not-config"

    assert exp1_command in mapping
    assert exp2_command in mapping
    assert other_command in mapping

    # Check ProcessInfo content
    assert mapping[exp1_command].pid == "12345"
    assert mapping[exp1_command].cmd == exp1_command
    assert mapping[exp2_command].pid == "54321"


def test_build_command_to_process_mapping_empty_gpu_data():
    """Test building command mapping with empty GPU data."""
    connected_gpus = []
    mapping = Manager.build_command_to_process_mapping(connected_gpus)
    assert len(mapping) == 0
    assert mapping == {}


def test_build_command_to_process_mapping_no_processes():
    """Test building command mapping when no processes are present."""
    connected_gpus = [
        GPUStatus(
            server='test_server',
            index=0,
            window_size=10,
            max_memory=0,
            processes=[
                ProcessInfo(
                    pid='12345',
                    user='testuser',
                    cmd='some_other_process --not-matching',
                    start_time='Mon Jan  1 10:00:00 2024',
                )
            ],
        )
    ]

    mapping = Manager.build_command_to_process_mapping(connected_gpus)
    assert len(mapping) == 1
    assert list(mapping.keys()) == ['some_other_process --not-matching']


def test_build_command_to_process_mapping_includes_cpu_processes():
    """CPU monitor processes should be included in the command mapping."""
    cpu_only_command = 'python project/scripts/run_cpu_job.py --scene demo'
    cpu_status = CPUStatus(
        server='cpu_server',
        window_size=10,
        processes=[
            ProcessInfo(
                pid='22222',
                user='cpuuser',
                cmd=cpu_only_command,
                start_time='Mon Jan  1 14:00:00 2024',
            )
        ],
        connected=True,
    )

    mapping = Manager.build_command_to_process_mapping([], [cpu_status])
    assert len(mapping) == 1
    assert cpu_only_command in mapping
    assert mapping[cpu_only_command].pid == '22222'


def test_build_command_to_process_mapping_deduplicates_gpu_and_cpu_records():
    """If GPU and CPU report the same command, metadata must match."""
    shared_command = 'python project/scripts/run_cpu_job.py --scene demo'
    gpu_process = ProcessInfo(
        pid='33333',
        user='shared',
        cmd=shared_command,
        start_time='Mon Jan  1 15:00:00 2024',
    )
    connected_gpus = [
        GPUStatus(
            server='gpu_server',
            index=0,
            window_size=10,
            max_memory=0,
            processes=[gpu_process],
        )
    ]

    cpu_status = CPUStatus(
        server='gpu_server',
        window_size=10,
        processes=[
            ProcessInfo(
                pid='33333',
                user='shared',
                cmd=shared_command,
                start_time='Mon Jan  1 15:00:00 2024',
            )
        ],
        connected=True,
    )

    with pytest.raises(AssertionError, match="Duplicate process PID '33333' detected on gpu_server"):
        Manager.build_command_to_process_mapping(connected_gpus, [cpu_status])
