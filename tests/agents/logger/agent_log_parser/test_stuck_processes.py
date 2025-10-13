import pytest
from agents.logger import AgentLogParser


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
    result = parser._parse_stuck_processes(
        "config1.py: (server1, 12345), config2.py: (server2, 67890)"
    )

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
