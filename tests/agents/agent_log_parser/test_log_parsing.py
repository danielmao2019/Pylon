from datetime import datetime
import pytest
from agents.logger import AgentLogParser


# ============================================================================
# LOG PARSING TESTS
# ============================================================================

def test_extract_key_events_from_nonexistent_file():
    """Test extraction from nonexistent file returns empty list."""
    parser = AgentLogParser(agent_log_path="/nonexistent/file.log")
    events = parser.extract_key_events_since_yesterday()
    assert events == []


def test_extract_key_events_since_yesterday(temp_agent_log, sample_log_entries):
    """Test extraction of key events from yesterday."""
    # Write sample log entries to temporary file
    with open(temp_agent_log, 'w') as f:
        for entry in sample_log_entries:
            f.write(entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    # Should extract events from yesterday and today, but not older
    assert len(events) >= 4  # At least stuck_removal, outdated_cleanup, error_message, ssh_launch
    
    # Check that events are sorted by timestamp
    timestamps = [event['timestamp'] for event in events]
    assert timestamps == sorted(timestamps)
    
    # Verify event types are present
    event_types = {event['type'] for event in events}
    expected_types = {'stuck_removal', 'outdated_cleanup', 'experiment_error', 'job_launch', 'critical_error'}
    assert event_types.intersection(expected_types) == expected_types


def test_parse_stuck_removal_event(temp_agent_log):
    """Test parsing of stuck process removal events."""
    log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} The following processes will be killed {{config1.py: (server1, 12345), config2.py: (server2, 67890)}}"
    
    with open(temp_agent_log, 'w') as f:
        f.write(log_entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    stuck_events = [e for e in events if e['type'] == 'stuck_removal']
    assert len(stuck_events) == 1
    
    event = stuck_events[0]
    assert event['type'] == 'stuck_removal'
    assert 'Stuck processes removed' in event['message']
    assert 'details' in event
    assert event['details']['process_count'] == 2
    assert 'config1.py' in event['details']['processes']
    assert event['details']['processes']['config1.py']['server'] == 'server1'
    assert event['details']['processes']['config1.py']['pid'] == '12345'


def test_parse_outdated_cleanup_event(temp_agent_log):
    """Test parsing of outdated cleanup events."""
    log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} The following runs has not been updated in the last 30 days and will be removed"
    
    with open(temp_agent_log, 'w') as f:
        f.write(log_entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    cleanup_events = [e for e in events if e['type'] == 'outdated_cleanup']
    assert len(cleanup_events) == 1
    
    event = cleanup_events[0]
    assert event['type'] == 'outdated_cleanup'
    assert 'Cleaned outdated runs' in event['message']
    assert event['details']['days_threshold'] == 30


def test_parse_experiment_error_event(temp_agent_log):
    """Test parsing of experiment error events."""
    log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Please fix model_v3.py. error_log='CUDA out of memory: Tried to allocate 2.00 GiB'"
    
    with open(temp_agent_log, 'w') as f:
        f.write(log_entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    error_events = [e for e in events if e['type'] == 'experiment_error']
    assert len(error_events) == 1
    
    event = error_events[0]
    assert event['type'] == 'experiment_error'
    assert 'Experiment failed: model_v3.py' in event['message']
    assert event['details']['config_file'] == 'model_v3.py'
    assert 'CUDA out of memory' in event['details']['error_log']


def test_parse_job_launch_event(temp_agent_log):
    """Test parsing of SSH job launch events."""
    log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ssh user@server1 'cd /path && python main.py --config-filepath configs/exp/baseline.py'"
    
    with open(temp_agent_log, 'w') as f:
        f.write(log_entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    launch_events = [e for e in events if e['type'] == 'job_launch']
    assert len(launch_events) == 1
    
    event = launch_events[0]
    assert event['type'] == 'job_launch'
    assert 'Launched experiment: baseline.py' in event['message']
    assert event['details']['config_path'] == 'configs/exp/baseline.py'


def test_parse_critical_error_event(temp_agent_log):
    """Test parsing of critical system error events."""
    log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} SSH connection failed to server2: Connection timeout"
    
    with open(temp_agent_log, 'w') as f:
        f.write(log_entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    critical_events = [e for e in events if e['type'] == 'critical_error']
    assert len(critical_events) == 1
    
    event = critical_events[0]
    assert event['type'] == 'critical_error'
    assert 'Critical system error detected' in event['message']
    assert 'SSH connection failed' in event['details']['error_line']


def test_malformed_log_entries(temp_agent_log):
    """Test handling of malformed log entries."""
    malformed_entries = [
        "Entry without timestamp",
        "2024-13-45 Invalid timestamp format",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Valid timestamp but no matching pattern"
    ]
    
    with open(temp_agent_log, 'w') as f:
        for entry in malformed_entries:
            f.write(entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    # Should handle malformed entries gracefully
    assert isinstance(events, list)


def test_empty_log_file(temp_agent_log):
    """Test parsing empty log file."""
    # Create empty file
    with open(temp_agent_log, 'w') as f:
        pass
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    assert events == []


def test_long_error_message_truncation(temp_agent_log):
    """Test that long error messages are properly truncated."""
    long_error = "A" * 300  # Error longer than 200 chars
    log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Please fix model.py. error_log='{long_error}'"
    
    with open(temp_agent_log, 'w') as f:
        f.write(log_entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    error_events = [e for e in events if e['type'] == 'experiment_error']
    assert len(error_events) == 1
    
    error_log = error_events[0]['details']['error_log']
    assert len(error_log) <= 203  # 200 + "..."
    assert error_log.endswith('...')


@pytest.mark.parametrize("critical_keyword", [
    "ssh connection failed",
    "SystemMonitor disconnected", 
    "Agent crash",
    "disk full",
    "permission denied"
])
def test_critical_error_keywords(temp_agent_log, critical_keyword):
    """Test that various critical error keywords are detected."""
    log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} System error: {critical_keyword} encountered"
    
    with open(temp_agent_log, 'w') as f:
        f.write(log_entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    critical_events = [e for e in events if e['type'] == 'critical_error']
    assert len(critical_events) == 1
