"""
Shared test fixtures for automation tests.

This conftest.py provides common fixtures used across multiple automation test modules
to eliminate duplication and ensure consistency.
"""
import pytest


# ============================================================================
# COMMON TEST CONSTANTS AS FIXTURES
# ============================================================================

@pytest.fixture
def EXPECTED_FILES():
    """Fixture that provides standard expected files for most tests."""
    return ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
