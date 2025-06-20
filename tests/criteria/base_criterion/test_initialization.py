"""Test initialization patterns for BaseCriterion."""
import pytest
from .conftest import ConcreteCriterion


def test_initialization_with_buffer():
    """Test initialization with buffer enabled (default)."""
    criterion = ConcreteCriterion()
    
    assert criterion.use_buffer is True
    assert hasattr(criterion, '_buffer_lock')
    assert hasattr(criterion, '_buffer_queue')
    assert hasattr(criterion, '_buffer_thread')
    assert hasattr(criterion, 'buffer')
    assert isinstance(criterion.buffer, list)
    assert len(criterion.buffer) == 0
    assert criterion._buffer_thread.is_alive()
    assert criterion._buffer_thread.daemon is True


def test_initialization_without_buffer():
    """Test initialization with buffer disabled."""
    criterion = ConcreteCriterion(use_buffer=False)
    
    assert criterion.use_buffer is False
    assert not hasattr(criterion, '_buffer_lock')
    assert not hasattr(criterion, '_buffer_queue')
    assert not hasattr(criterion, '_buffer_thread')
    assert not hasattr(criterion, 'buffer')


@pytest.mark.parametrize("use_buffer", [True, False])
def test_parameter_validation(use_buffer):
    """Test parameter validation during initialization."""
    criterion = ConcreteCriterion(use_buffer=use_buffer)
    assert criterion.use_buffer is use_buffer


def test_abstract_methods_not_implemented():
    """Test that abstract methods raise NotImplementedError."""
    from criteria.base_criterion import BaseCriterion
    
    # Cannot instantiate abstract base class directly
    with pytest.raises(TypeError):
        BaseCriterion()


def test_state_initialization():
    """Test internal state initialization."""
    criterion = ConcreteCriterion()
    
    # Test custom state variables
    assert criterion.call_count == 0
    assert criterion.summarize_count == 0
    
    # Test buffer state
    assert len(criterion.buffer) == 0