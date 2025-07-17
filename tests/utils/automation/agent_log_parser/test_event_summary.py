import pytest
from utils.automation.agent_log_parser import AgentLogParser


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