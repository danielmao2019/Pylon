from agents.manager import Manager
from agents.monitor.gpu_status import GPUStatus
from agents.monitor.process_info import ProcessInfo


def test_manager_build_config_to_process_mapping():
    """Test building config to ProcessInfo mapping from GPU data."""
    connected_gpus = [
        GPUStatus(
            server='test_server', index=0, window_size=10, max_memory=0,
            processes=[
                ProcessInfo(
                    pid='12345',
                    user='testuser',
                    cmd='python main.py --config-filepath configs/exp1.py',
                    start_time='Mon Jan  1 10:00:00 2024'
                ),
                ProcessInfo(
                    pid='12346',
                    user='testuser', 
                    cmd='some_other_process --not-config',
                    start_time='Mon Jan  1 11:00:00 2024'
                )
            ]
        ),
        GPUStatus(
            server='test_server', index=1, window_size=10, max_memory=0,
            processes=[
                ProcessInfo(
                    pid='54321',
                    user='testuser',
                    cmd='python main.py --config-filepath configs/exp2.py --debug',
                    start_time='Mon Jan  1 12:00:00 2024'
                )
            ]
        ),
    ]
    
    mapping = Manager.build_config_to_process_mapping(connected_gpus)
    
    # Should only include processes with python main.py --config-filepath
    assert len(mapping) == 2
    assert "configs/exp1.py" in mapping
    assert "configs/exp2.py" in mapping
    
    # Check ProcessInfo content
    assert mapping["configs/exp1.py"].pid == "12345"
    assert mapping["configs/exp1.py"].cmd == "python main.py --config-filepath configs/exp1.py"
    assert mapping["configs/exp2.py"].pid == "54321"


def test_build_config_to_process_mapping_empty_gpu_data():
    """Test building config mapping with empty GPU data."""
    connected_gpus = []
    mapping = Manager.build_config_to_process_mapping(connected_gpus)
    assert len(mapping) == 0
    assert mapping == {}


def test_build_config_to_process_mapping_no_matching_processes():
    """Test building config mapping when no processes match pattern."""
    connected_gpus = [
        GPUStatus(
            server='test_server', index=0, window_size=10, max_memory=0,
            processes=[
                ProcessInfo(
                    pid='12345',
                    user='testuser',
                    cmd='some_other_process --not-matching',
                    start_time='Mon Jan  1 10:00:00 2024'
                )
            ]
        )
    ]
    
    mapping = Manager.build_config_to_process_mapping(connected_gpus)
    assert len(mapping) == 0
    assert mapping == {}

