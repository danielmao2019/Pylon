import os
import json
import tempfile
import shutil
import time
from datetime import datetime
from unittest.mock import Mock
import pytest
from agents.logger import LogsSnapshot
from agents.monitor.system_monitor import SystemMonitor
from agents.manager import Manager
from agents.manager.base_job import BaseJob
from agents.manager.base_progress_info import BaseProgressInfo
from agents.manager.default_job import DefaultJobProgressInfo
from agents.monitor.process_info import ProcessInfo
from agents.monitor.gpu_status import GPUStatus
from utils.io.json import serialize_object

BASE_COMMAND = 'python main.py --config-filepath configs/exp/baseline.py'
SAMPLE_COMMANDS = [
    BASE_COMMAND,
    'python main.py --config-filepath configs/exp/model_v2.py',
    'python main.py --config-filepath configs/exp/ablation.py',
]


def _build_dummy_job(command: str, progress: BaseProgressInfo, process_info: ProcessInfo) -> BaseJob:
    class _DummyJob(BaseJob):
        def derive_work_dir(self) -> str:
            return './logs/baseline_run'

        def compute_progress(self, runtime):
            return progress

        def is_active(self, runtime):
            return True

        def is_complete(self, progress_value, runtime):
            return False

        def is_stuck(self, runtime):
            return False

        def tmux_session_name(self) -> str:
            return 'dummy'

    job = _DummyJob(command)
    job.progress = progress
    job.status = 'running'
    job.process_info = process_info
    return job


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_snapshot_dir():
    """Create a temporary directory for snapshots."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config_files():
    """Sample config files for testing."""
    return SAMPLE_COMMANDS


@pytest.fixture
def sample_expected_files():
    """Sample expected files for testing."""
    return ["train_metrics.json", "val_metrics.json", "model.pt"]


@pytest.fixture
def mock_system_monitor():
    """Create a mock SystemMonitor for testing."""
    mock_monitor = Mock(spec=SystemMonitor)
    mock_monitor.server = 'server1'
    mock_monitor.connected_gpus = [
        GPUStatus(
            server='server1',
            index=0,
            window_size=10,
            processes=[
                ProcessInfo(
                    pid='12345',
                    user='testuser',
                    cmd=BASE_COMMAND,
                    start_time='2025-07-17 10:00:00',
                )
            ],
            connected=True,
        )
    ]
    return mock_monitor


@pytest.fixture
def sample_progress_info():
    """Sample progress information for testing."""
    return DefaultJobProgressInfo(
        completed_epochs=15,
        progress_percentage=75.0,
        early_stopped=False,
        early_stopped_at_epoch=None,
    )


@pytest.fixture
def sample_process_info():
    """Sample ProcessInfo for testing."""
    return ProcessInfo(
        pid='12345',
        user='testuser',
        cmd=BASE_COMMAND,
        start_time='2025-07-17 10:00:00',
    )


@pytest.fixture
def sample_job_status(sample_progress_info, sample_process_info):
    """Sample BaseJob for testing."""
    job = _build_dummy_job(BASE_COMMAND, sample_progress_info, sample_process_info)
    return job.to_dict()


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_logs_snapshot_initialization(sample_config_files):
    """Test LogsSnapshot initialization with valid parameters."""
    snapshot = LogsSnapshot(
        commands=sample_config_files, epochs=20, sleep_time=3600, outdated_days=15
    )

    assert snapshot.commands == sample_config_files
    # expected_files are handled by downstream utilities; not part of LogsSnapshot API
    assert snapshot.epochs == 20
    assert snapshot.sleep_time == 3600
    assert snapshot.outdated_days == 15
    assert snapshot.snapshot_dir == "./agents/snapshots"


def test_logs_snapshot_initialization_with_defaults(sample_config_files):
    """Test LogsSnapshot initialization with default parameters."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)

    assert snapshot.sleep_time == 86400  # Default
    assert snapshot.outdated_days == 30  # Default


def test_logs_snapshot_initialization_validation():
    """Test LogsSnapshot initialization parameter validation."""
    # Test invalid config_files type
    with pytest.raises(AssertionError, match="commands must be list"):
        LogsSnapshot(commands="not_a_list", epochs=20)

    # expected_files is not an argument to LogsSnapshot

    # Test invalid epochs type
    with pytest.raises(AssertionError, match="epochs must be int"):
        LogsSnapshot(commands=[BASE_COMMAND], epochs="20")


# ============================================================================
# SNAPSHOT CREATION TESTS
# ============================================================================


def test_create_snapshot(sample_config_files, mock_system_monitor, monkeypatch, sample_progress_info, sample_process_info):
    """Test snapshot creation."""

    # Mock Manager.build_jobs to return test data
    def mock_build_jobs(self):
        job = _build_dummy_job(BASE_COMMAND, sample_progress_info, sample_process_info)
        return {BASE_COMMAND: job}

    monkeypatch.setattr(Manager, "build_jobs", mock_build_jobs)

    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)

    timestamp = "2025-07-17_123000"
    result = snapshot.create_snapshot(timestamp, mock_system_monitor)

    assert result['timestamp'] == timestamp
    assert 'job_statuses' in result
    assert 'snapshot_metadata' in result
    assert result['snapshot_metadata']['total_commands'] == len(sample_config_files)
    assert BASE_COMMAND in result['job_statuses']


def test_create_snapshot_parameter_validation(sample_config_files, mock_system_monitor):
    """Test create_snapshot parameter validation."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)

    # Test invalid timestamp type
    with pytest.raises(AssertionError, match="timestamp must be str"):
        snapshot.create_snapshot(123, mock_system_monitor)

    # Test invalid system_monitor type
    with pytest.raises(AssertionError, match="system_monitor must be SystemMonitor"):
        snapshot.create_snapshot("2025-07-17_123000", "not_a_monitor")


# ============================================================================
# SNAPSHOT SAVING AND LOADING TESTS
# ============================================================================


def test_save_and_load_snapshot(
    temp_snapshot_dir, sample_config_files, sample_expected_files, sample_job_status
):
    """Test saving and loading snapshots."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)
    snapshot.snapshot_dir = temp_snapshot_dir

    # Create test snapshot data
    snapshot_data = {
        'timestamp': '2025-07-17_123000',
        'job_statuses': {BASE_COMMAND: sample_job_status},
        'snapshot_metadata': {
            'total_commands': 1,
            'expected_files': sample_expected_files,
            'epochs': 20,
        },
    }

    filename = "test_snapshot.json"

    # Test saving
    snapshot.save_snapshot(snapshot_data, filename)

    # Verify file was created
    filepath = os.path.join(temp_snapshot_dir, filename)
    assert os.path.exists(filepath)

    # Test loading
    loaded_data = snapshot.load_snapshot(filename)
    assert loaded_data is not None
    assert loaded_data['timestamp'] == '2025-07-17_123000'
    assert 'job_statuses' in loaded_data
    assert 'snapshot_metadata' in loaded_data


def test_save_snapshot_parameter_validation(temp_snapshot_dir, sample_config_files):
    """Test save_snapshot parameter validation."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)
    snapshot.snapshot_dir = temp_snapshot_dir

    # Test invalid snapshot type
    with pytest.raises(AssertionError, match="snapshot must be dict"):
        snapshot.save_snapshot("not_a_dict", "test.json")

    # Test invalid filename type
    with pytest.raises(AssertionError, match="filename must be str"):
        snapshot.save_snapshot({}, 123)

    # Test invalid filename extension
    with pytest.raises(AssertionError, match="filename must end with .json"):
        snapshot.save_snapshot({}, "test.txt")


def test_load_nonexistent_snapshot(temp_snapshot_dir, sample_config_files):
    """Test loading nonexistent snapshot returns None."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)
    snapshot.snapshot_dir = temp_snapshot_dir

    result = snapshot.load_snapshot("nonexistent.json")
    assert result is None


def test_load_corrupted_snapshot(temp_snapshot_dir, sample_config_files):
    """Test loading corrupted snapshot returns None."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)
    snapshot.snapshot_dir = temp_snapshot_dir

    # Create corrupted JSON file
    corrupted_file = os.path.join(temp_snapshot_dir, "corrupted.json")
    with open(corrupted_file, 'w') as f:
        f.write("{ invalid json content")

    result = snapshot.load_snapshot("corrupted.json")
    assert result is None


# ============================================================================
# JSON SERIALIZATION TESTS
# ============================================================================


def test_generic_serialization(sample_job_status):
    """Test generic JSON serialization of snapshot data."""

    snapshot_data = {
        'timestamp': '2025-07-17_123000',
        'job_statuses': {BASE_COMMAND: sample_job_status},
        'snapshot_metadata': {'total_commands': 1},
    }

    serializable = serialize_object(snapshot_data)

    # Verify job_statuses are serialized as dicts (dataclasses converted to dict)
    serialized_job = serializable['job_statuses'][BASE_COMMAND]
    assert isinstance(serialized_job, dict)
    assert serialized_job['command'] == BASE_COMMAND
    assert 'work_dir' in serialized_job
    assert 'progress' in serialized_job
    assert 'process_info' in serialized_job

    # Verify it can be JSON serialized
    json_str = json.dumps(serializable)
    assert isinstance(json_str, str)


def test_generic_serialization_with_datetime():
    """Test generic JSON serialization with datetime objects."""

    snapshot_data = {'timestamp': datetime.now(), 'other_data': 'test'}

    serializable = serialize_object(snapshot_data)

    # Verify datetime is converted to ISO format
    assert isinstance(serializable['timestamp'], str)
    assert 'T' in serializable['timestamp']  # ISO format contains 'T'


# ============================================================================
# SNAPSHOT MANAGEMENT TESTS
# ============================================================================


def test_list_snapshots_empty_directory(temp_snapshot_dir, sample_config_files):
    """Test listing snapshots in empty directory."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)
    snapshot.snapshot_dir = temp_snapshot_dir

    snapshots = snapshot.list_snapshots()
    assert snapshots == []


def test_list_snapshots_with_files(temp_snapshot_dir, sample_config_files):
    """Test listing snapshots with existing files."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)
    snapshot.snapshot_dir = temp_snapshot_dir

    # Create test snapshot files
    test_files = ["snapshot1.json", "snapshot2.json", "not_snapshot.txt"]
    for filename in test_files:
        filepath = os.path.join(temp_snapshot_dir, filename)
        with open(filepath, 'w') as f:
            json.dump({'test': 'data'}, f)

    snapshots = snapshot.list_snapshots()

    # Should only include .json files
    assert len(snapshots) == 2
    assert "snapshot1.json" in snapshots
    assert "snapshot2.json" in snapshots
    assert "not_snapshot.txt" not in snapshots


def test_cleanup_old_snapshots(temp_snapshot_dir, sample_config_files):
    """Test cleanup of old snapshots."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)
    snapshot.snapshot_dir = temp_snapshot_dir

    # Create test files with different ages
    current_time = time.time()
    old_time = current_time - (35 * 24 * 60 * 60)  # 35 days old
    recent_time = current_time - (10 * 24 * 60 * 60)  # 10 days old

    old_file = os.path.join(temp_snapshot_dir, "old_snapshot.json")
    recent_file = os.path.join(temp_snapshot_dir, "recent_snapshot.json")

    # Create files
    with open(old_file, 'w') as f:
        json.dump({'test': 'old'}, f)
    with open(recent_file, 'w') as f:
        json.dump({'test': 'recent'}, f)

    # Set modification times
    os.utime(old_file, (old_time, old_time))
    os.utime(recent_file, (recent_time, recent_time))

    # Cleanup snapshots older than 30 days
    removed_count = snapshot.cleanup_old_snapshots(retention_days=30)

    assert removed_count == 1
    assert not os.path.exists(old_file)
    assert os.path.exists(recent_file)


def test_cleanup_old_snapshots_parameter_validation(sample_config_files):
    """Test cleanup_old_snapshots parameter validation."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)

    # Test invalid retention_days type
    with pytest.raises(AssertionError, match="retention_days must be int"):
        snapshot.cleanup_old_snapshots(retention_days="30")

    # Test invalid retention_days value
    with pytest.raises(AssertionError, match="retention_days must be positive"):
        snapshot.cleanup_old_snapshots(retention_days=0)


def test_get_snapshot_statistics_empty(temp_snapshot_dir, sample_config_files):
    """Test snapshot statistics for empty directory."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)
    snapshot.snapshot_dir = temp_snapshot_dir

    stats = snapshot.get_snapshot_statistics()

    assert stats['total_snapshots'] == 0
    assert stats['oldest_snapshot'] is None
    assert stats['newest_snapshot'] is None
    assert stats['total_size_bytes'] == 0


def test_get_snapshot_statistics_with_files(temp_snapshot_dir, sample_config_files):
    """Test snapshot statistics with existing files."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)
    snapshot.snapshot_dir = temp_snapshot_dir

    # Create test files
    test_data = {'test': 'data' * 100}  # Make files have some size
    filenames = ["snapshot1.json", "snapshot2.json"]

    for filename in filenames:
        filepath = os.path.join(temp_snapshot_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(test_data, f)

    stats = snapshot.get_snapshot_statistics()

    assert stats['total_snapshots'] == 2
    assert stats['oldest_snapshot'] is not None
    assert stats['newest_snapshot'] is not None
    assert stats['total_size_bytes'] > 0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_snapshot_directory_creation(sample_config_files):
    """Test that snapshot directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        nonexistent_dir = os.path.join(temp_dir, "nonexistent", "snapshots")

        snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)
        snapshot.snapshot_dir = nonexistent_dir

        # Save a snapshot - should create directory
        snapshot_data = {'timestamp': '2025-07-17_123000', 'job_statuses': {}}
        snapshot.save_snapshot(snapshot_data, "test.json")

        assert os.path.exists(nonexistent_dir)
        assert os.path.exists(os.path.join(nonexistent_dir, "test.json"))


def test_cleanup_nonexistent_directory(sample_config_files):
    """Test cleanup when snapshot directory doesn't exist."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)
    snapshot.snapshot_dir = "/nonexistent/directory"

    removed_count = snapshot.cleanup_old_snapshots()
    assert removed_count == 0


def test_load_snapshot_parameter_validation(sample_config_files):
    """Test load_snapshot parameter validation."""
    snapshot = LogsSnapshot(commands=sample_config_files, epochs=20)

    # Test invalid filename type
    with pytest.raises(AssertionError, match="filename must be str"):
        snapshot.load_snapshot(123)
