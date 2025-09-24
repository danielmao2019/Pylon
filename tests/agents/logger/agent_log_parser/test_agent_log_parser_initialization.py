import pytest
from agents.logger import AgentLogParser


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
