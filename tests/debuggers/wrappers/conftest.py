import pytest
import tempfile
import os
import shutil
from typing import Dict, Any


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def small_page_size():
    """Small page size for testing page management."""
    return 1024  # 1KB for quick page turnover in tests


@pytest.fixture
def large_debug_outputs():
    """Large debug outputs to trigger page management."""
    import torch
    return {
        'large_tensor': torch.randn(1000, 1000),  # Large tensor to trigger page saving
        'metadata': {'size': 1000000}
    }


@pytest.fixture
def multiple_datapoints():
    """Multiple sample datapoints for buffer testing."""
    import torch

    datapoints = []
    for i in range(5):
        datapoint = {
            'inputs': torch.randn(2, 3, 32, 32, dtype=torch.float32),
            'labels': torch.randint(0, 10, (2,), dtype=torch.int64),
            'outputs': torch.randn(2, 10, dtype=torch.float32),
            'meta_info': {
                'idx': [i],
                'image_path': [f'/path/to/image_{i}.jpg'],
            }
        }
        datapoints.append(datapoint)

    return datapoints