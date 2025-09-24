"""
Shared test fixtures for automation tests.

This conftest.py provides common fixtures used across multiple automation test modules
to eliminate duplication and ensure consistency.
"""
from typing import Optional, Dict, List
import os
import json
import pytest
import torch
from unittest.mock import Mock
from utils.monitor.system_monitor import SystemMonitor
from utils.io.json import save_json


# ============================================================================
# COMMON TEST CONSTANTS AS FIXTURES
# ============================================================================

@pytest.fixture
def EXPECTED_FILES():
    """Fixture that provides standard expected files for most tests."""
    return ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]


# ============================================================================
# COMMON TEST DATA CREATION HELPERS
# ============================================================================

# Moved to tests/agents/tracker/conftest.py
# moved to tests/agents/tracker/conftest.py


# Moved to tests/agents/tracker/conftest.py
# moved to tests/agents/tracker/conftest.py


# Moved to tests/agents/tracker/conftest.py
# moved to tests/agents/tracker/conftest.py


@pytest.fixture
def create_minimal_system_monitor_with_processes():
    """Fixture that provides function to create a minimal SystemMonitor mock with realistic data.

    This is the ONLY necessary mock in our test suite because:
    1. SystemMonitor has complex internal state and SSH dependencies
    2. We only need the connected_gpus property for our testing
    3. All other data structures are real (ProcessInfo, RunStatus, etc.)

    This mock only mocks the SystemMonitor interface, not the data.

    Returns:
        Function(connected_gpus_data: List[Dict]) -> Mock
    """
    def _create_minimal_system_monitor_with_processes(connected_gpus_data: List[Dict]) -> Mock:
        mock_monitor = Mock(spec=SystemMonitor)
        mock_monitor.connected_gpus = connected_gpus_data
        return mock_monitor

    return _create_minimal_system_monitor_with_processes


@pytest.fixture
def setup_realistic_experiment_structure(create_real_config, create_epoch_files, create_minimal_system_monitor_with_processes):
    """Fixture that provides function to set up a realistic experiment directory structure.

    Args:
        temp_root: Root temporary directory
        experiments: List of (exp_name, target_status, epochs_completed, has_recent_logs) tuples

    Returns:
        Function(temp_root: str, experiments: List[tuple]) -> tuple of (config_files, work_dirs, system_monitor)
    """
    def _setup_realistic_experiment_structure(temp_root: str, experiments: List[tuple]) -> tuple:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")

        config_files = []
        work_dirs = {}
        running_experiments = []

        for exp_name, target_status, epochs_completed, has_recent_logs in experiments:
            work_dir = os.path.join(logs_dir, exp_name)
            config_path = os.path.join(configs_dir, f"{exp_name}.py")

            os.makedirs(work_dir, exist_ok=True)
            create_real_config(config_path, work_dir, epochs=100)
            config_files.append(config_path)
            work_dirs[config_path] = work_dir

            # Create epoch files
            for epoch_idx in range(epochs_completed):
                create_epoch_files(work_dir, epoch_idx)

            # Create recent logs if needed
            if has_recent_logs:
                log_file = os.path.join(work_dir, "train_val_latest.log")
                with open(log_file, 'w') as f:
                    f.write("Recent training log")

            # Track which experiments should be running on GPU
            if target_status in ["running", "stuck"]:
                running_experiments.append(config_path)

        # Create realistic GPU process data
        from utils.monitor.process_info import ProcessInfo
        connected_gpus_data = [
            {
                'server': 'test_server',
                'index': 0,
                'processes': [
                    ProcessInfo(
                        pid=f'1234{i}',
                        user='testuser',
                        cmd=f'python main.py --config-filepath {config_path}',
                        start_time=f'Mon Jan  1 1{i}:00:00 2024'
                    )
                    for i, config_path in enumerate(running_experiments)
                ]
            }
        ] if running_experiments else [{'server': 'test_server', 'index': 0, 'processes': []}]

        # Create minimal SystemMonitor mock with realistic data
        system_monitor = create_minimal_system_monitor_with_processes(connected_gpus_data)

        return config_files, work_dirs, system_monitor

    return _setup_realistic_experiment_structure
