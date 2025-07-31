import pytest
from data.cache.cpu_dataset_cache import CPUDatasetCache


def test_default_initialization():
    """Test cache initialization with default parameters."""
    cache = CPUDatasetCache()
    assert cache.max_memory_percent == 80.0
    assert cache.enable_validation is True

    # Test initial state
    assert len(cache.cache) == 0
    assert len(cache.checksums) == 0


def test_custom_initialization():
    """Test cache initialization with custom parameters."""
    cache = CPUDatasetCache(max_memory_percent=50.0, enable_validation=False)
    assert cache.max_memory_percent == 50.0
    assert cache.enable_validation is False


@pytest.mark.parametrize("invalid_percent,error_msg", [
    (-1.0, "must be between 0 and 100"),
    (101.0, "must be between 0 and 100"),
])
def test_invalid_initialization(invalid_percent, error_msg):
    """Test initialization with invalid parameters."""
    with pytest.raises(ValueError, match=error_msg):
        CPUDatasetCache(max_memory_percent=invalid_percent)
