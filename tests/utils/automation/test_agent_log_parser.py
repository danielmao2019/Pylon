from typing import List, Dict, Any
import os
import json
import tempfile
from datetime import datetime, timedelta
import pytest
from utils.automation.agent_log_parser import AgentLogParser


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_agent_log():
    """Create a temporary agent log file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_log_entries():
    """Sample log entries for testing parsing."""
    now = datetime.now()
    recent_time = now - timedelta(hours=1)  # Use recent time within yesterday
    
    return [
        # Stuck process removal
        f"{recent_time.strftime('%Y-%m-%d %H:%M:%S')} The following processes will be killed {{config1.py: (server1, 12345), config2.py: (server2, 67890)}}",
        
        # Outdated cleanup
        f"{recent_time.strftime('%Y-%m-%d %H:%M:%S')} The following runs has not been updated in the last 30 days and will be removed",
        
        # Error message
        f"{recent_time.strftime('%Y-%m-%d %H:%M:%S')} Please fix model_v3.py. error_log='CUDA out of memory: Tried to allocate 2.00 GiB'",
        
        # SSH job launch
        f"{now.strftime('%Y-%m-%d %H:%M:%S')} ssh user@server1 'cd /path && python main.py --config-filepath configs/exp/baseline.py'",
        
        # Critical error
        f"{recent_time.strftime('%Y-%m-%d %H:%M:%S')} SSH connection failed to server2: Connection timeout",
        
        # Entry without timestamp (should be ignored)
        "This line has no timestamp and should be ignored",
        
        # Entry too old (should be filtered out)
        f"{(datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')} Old entry that should be filtered out"
    ]


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_agent_log_parser_initialization():
    """Test AgentLogParser initialization with default and custom paths."""
    # Test default path
    parser = AgentLogParser()
    assert parser.agent_log_path == "./project/run_agent.log"
    assert 'stuck_removal' in parser.patterns
    assert 'outdated_cleanup' in parser.patterns
    assert 'error_message' in parser.patterns
    assert 'ssh_launch' in parser.patterns
    assert 'timestamp' in parser.patterns
    
    # Test custom path
    custom_path = "/custom/path/agent.log"
    parser = AgentLogParser(agent_log_path=custom_path)
    assert parser.agent_log_path == custom_path


def test_regex_patterns_compiled():
    """Test that all regex patterns are properly compiled."""
    parser = AgentLogParser()
    
    for pattern_name, pattern in parser.patterns.items():
        assert hasattr(pattern, 'search'), f"Pattern {pattern_name} should be compiled regex"


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


# ============================================================================
# TIME FILTERING TESTS
# ============================================================================

def test_time_based_filtering(temp_agent_log):
    """Test that only events from yesterday onward are included."""
    yesterday = datetime.now() - timedelta(days=1)
    old_date = datetime.now() - timedelta(days=3)
    
    log_entries = [
        f"{old_date.strftime('%Y-%m-%d %H:%M:%S')} Old event should be filtered",
        f"{yesterday.strftime('%Y-%m-%d %H:%M:%S')} The following processes will be killed {{config1.py: (server1, 12345)}}",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Recent event should be included"
    ]
    
    with open(temp_agent_log, 'w') as f:
        for entry in log_entries:
            f.write(entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    # Should only include events from yesterday onward
    for event in events:
        assert event['timestamp'] >= yesterday


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

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


def test_log_parsing_error_handling(temp_agent_log):
    """Test error handling when log file becomes corrupted during reading."""
    with open(temp_agent_log, 'w') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Normal entry\n")
    
    # Make file unreadable after creation
    os.chmod(temp_agent_log, 0o000)
    
    try:
        parser = AgentLogParser(agent_log_path=temp_agent_log)
        events = parser.extract_key_events_since_yesterday()
        
        # Should return parser error event
        error_events = [e for e in events if e['type'] == 'parser_error']
        assert len(error_events) == 1
        assert 'Error parsing agent log' in error_events[0]['message']
    finally:
        # Restore permissions for cleanup
        os.chmod(temp_agent_log, 0o644)


# ============================================================================
# STUCK PROCESSES PARSING TESTS
# ============================================================================

def test_parse_stuck_processes_single():
    """Test parsing single stuck process."""
    parser = AgentLogParser()
    result = parser._parse_stuck_processes("config1.py: (server1, 12345)")
    
    assert result['process_count'] == 1
    assert 'config1.py' in result['processes']
    assert result['processes']['config1.py']['server'] == 'server1'
    assert result['processes']['config1.py']['pid'] == '12345'


def test_parse_stuck_processes_multiple():
    """Test parsing multiple stuck processes."""
    parser = AgentLogParser()
    result = parser._parse_stuck_processes("config1.py: (server1, 12345), config2.py: (server2, 67890)")
    
    assert result['process_count'] == 2
    assert 'config1.py' in result['processes']
    assert 'config2.py' in result['processes']
    assert result['processes']['config1.py']['server'] == 'server1'
    assert result['processes']['config2.py']['server'] == 'server2'


def test_parse_stuck_processes_malformed():
    """Test parsing malformed stuck processes string."""
    parser = AgentLogParser()
    result = parser._parse_stuck_processes("malformed string without proper format")
    
    # Should handle gracefully and return basic info
    assert result['process_count'] == 0
    assert 'raw_info' in result


# ============================================================================
# EVENT SUMMARY TESTS
# ============================================================================

def test_get_event_summary_empty():
    """Test event summary with no events."""
    parser = AgentLogParser(agent_log_path="/nonexistent/file.log")
    summary = parser.get_event_summary_since_yesterday()
    
    assert summary['total_events'] == 0
    assert summary['event_counts'] == {}
    assert summary['critical_issues'] == 0
    assert summary['experiments_affected'] == 0
    assert summary['time_range']['start'] is None
    assert summary['time_range']['end'] is None


def test_get_event_summary_with_events(temp_agent_log, sample_log_entries):
    """Test event summary with actual events."""
    with open(temp_agent_log, 'w') as f:
        for entry in sample_log_entries:
            f.write(entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    summary = parser.get_event_summary_since_yesterday()
    
    assert summary['total_events'] > 0
    assert len(summary['event_counts']) > 0
    assert summary['critical_issues'] >= 2  # experiment_error + critical_error
    assert summary['experiments_affected'] > 0
    assert summary['time_range']['start'] is not None
    assert summary['time_range']['end'] is not None


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

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
