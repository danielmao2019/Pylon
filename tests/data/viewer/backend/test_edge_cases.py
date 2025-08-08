"""Tests for ViewerBackend edge cases and error handling.

This module tests error conditions, input validation, and edge cases
using pytest.raises patterns following Pylon testing philosophy.
"""

from typing import Dict, Any, List, Tuple

import pytest
import torch

from data.datasets.base_dataset import BaseDataset
from data.viewer.backend.backend import ViewerBackend


@pytest.fixture
def backend():
    """Create a ViewerBackend instance for testing."""
    return ViewerBackend()


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for error testing."""
    
    class _MockDataset(BaseDataset):
        SPLIT_OPTIONS = ["test"]
        INPUT_NAMES = ["data"]
        LABEL_NAMES = ["label"]

        def _init_annotations(self) -> None:
            self.annotations = [{"id": 0}, {"id": 1}]

        def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
            data = torch.randn(10, dtype=torch.float32)
            label = torch.randint(0, 5, (1,), dtype=torch.int64)
            return {"data": data}, {"label": label}, {"sample_idx": idx}
        
        @staticmethod
        def display_datapoint(datapoint, class_labels=None, camera_state=None, settings_3d=None):
            """Mock display method."""
            return None
    
    return _MockDataset(split="test", device=torch.device("cpu"))


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_get_dataset_instance_invalid_input_types(backend):
    """Test get_dataset_instance with invalid input types."""
    # Test with integer instead of string
    with pytest.raises(AssertionError) as exc_info:
        backend.get_dataset_instance(123)
    assert "dataset_name must be str" in str(exc_info.value)
    
    # Test with None
    with pytest.raises(AssertionError) as exc_info:
        backend.get_dataset_instance(None)
    assert "dataset_name must be str" in str(exc_info.value)
    
    # Test with list
    with pytest.raises(AssertionError) as exc_info:
        backend.get_dataset_instance(["invalid", "list"])
    assert "dataset_name must be str" in str(exc_info.value)
    
    # Test with dict
    with pytest.raises(AssertionError) as exc_info:
        backend.get_dataset_instance({"invalid": "dict"})
    assert "dataset_name must be str" in str(exc_info.value)


def test_get_dataset_instance_nonexistent_dataset(backend):
    """Test get_dataset_instance with nonexistent dataset."""
    with pytest.raises(ValueError) as exc_info:
        backend.get_dataset_instance("nonexistent/dataset")
    assert "Dataset not loaded: nonexistent/dataset" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        backend.get_dataset_instance("another/missing/dataset")
    assert "Dataset not loaded: another/missing/dataset" in str(exc_info.value)


def test_get_datapoint_invalid_dataset_name_type(backend):
    """Test get_datapoint with invalid dataset_name type."""
    with pytest.raises(AssertionError) as exc_info:
        backend.get_datapoint(123, 0, [])
    assert "dataset_name must be str" in str(exc_info.value)
    
    with pytest.raises(AssertionError) as exc_info:
        backend.get_datapoint(None, 0, [])
    assert "dataset_name must be str" in str(exc_info.value)


def test_get_datapoint_invalid_index_type(backend):
    """Test get_datapoint with invalid index type."""
    with pytest.raises(AssertionError) as exc_info:
        backend.get_datapoint("test/dataset", "invalid_index", [])
    assert "index must be int" in str(exc_info.value)
    
    with pytest.raises(AssertionError) as exc_info:
        backend.get_datapoint("test/dataset", 1.5, [])
    assert "index must be int" in str(exc_info.value)
    
    with pytest.raises(AssertionError) as exc_info:
        backend.get_datapoint("test/dataset", None, [])
    assert "index must be int" in str(exc_info.value)


def test_get_datapoint_invalid_transform_indices_type(backend):
    """Test get_datapoint with invalid transform_indices type."""
    with pytest.raises(AssertionError) as exc_info:
        backend.get_datapoint("test/dataset", 0, "not_a_list")
    assert "transform_indices must be list" in str(exc_info.value)
    
    with pytest.raises(AssertionError) as exc_info:
        backend.get_datapoint("test/dataset", 0, None)
    assert "transform_indices must be list" in str(exc_info.value)
    
    with pytest.raises(AssertionError) as exc_info:
        backend.get_datapoint("test/dataset", 0, 123)
    assert "transform_indices must be list" in str(exc_info.value)


def test_get_datapoint_invalid_transform_indices_content(backend):
    """Test get_datapoint with invalid transform_indices content."""
    # Test with non-integer elements
    with pytest.raises(AssertionError) as exc_info:
        backend.get_datapoint("test/dataset", 0, ["not_int", 1])
    assert "All transform indices must be int" in str(exc_info.value)
    
    with pytest.raises(AssertionError) as exc_info:
        backend.get_datapoint("test/dataset", 0, [0, 1.5, 2])
    assert "All transform indices must be int" in str(exc_info.value)
    
    with pytest.raises(AssertionError) as exc_info:
        backend.get_datapoint("test/dataset", 0, [0, None, 2])
    assert "All transform indices must be int" in str(exc_info.value)


def test_get_datapoint_nonexistent_dataset(backend):
    """Test get_datapoint with nonexistent dataset."""
    with pytest.raises(ValueError) as exc_info:
        backend.get_datapoint("nonexistent/dataset", 0, [])
    assert "Dataset not loaded: nonexistent/dataset" in str(exc_info.value)


def test_get_datapoint_out_of_bounds_index(backend, mock_dataset):
    """Test get_datapoint with out of bounds index."""
    # Store dataset in backend
    dataset_name = "test/MockDataset"
    backend._datasets[dataset_name] = mock_dataset
    
    # Test with negative index (might be valid in Python)
    # Test with index beyond dataset length
    dataset_length = len(mock_dataset)
    
    with pytest.raises(IndexError):
        backend.get_datapoint(dataset_name, dataset_length, [])
    
    with pytest.raises(IndexError):
        backend.get_datapoint(dataset_name, dataset_length + 10, [])


def test_get_datapoint_invalid_transform_index(backend, mock_dataset):
    """Test get_datapoint with invalid transform index."""
    # Store dataset in backend
    dataset_name = "test/MockDataset"
    backend._datasets[dataset_name] = mock_dataset
    
    # Try to use transform index when no transforms are registered
    with pytest.raises(IndexError):
        backend.get_datapoint(dataset_name, 0, [0])
    
    with pytest.raises(IndexError):
        backend.get_datapoint(dataset_name, 0, [1, 2, 3])


def test_load_dataset_nonexistent_config(backend):
    """Test load_dataset with nonexistent dataset configuration."""
    with pytest.raises(ValueError) as exc_info:
        backend.load_dataset("nonexistent/dataset")
    assert "Dataset not found: nonexistent/dataset" in str(exc_info.value)


def test_load_dataset_invalid_config_path(backend):
    """Test load_dataset with invalid config path."""
    # Add config with invalid path
    backend._configs["test/invalid"] = {
        'path': '/nonexistent/path/invalid_config.py',
        'type': 'test',
        'name': 'invalid'
    }
    
    with pytest.raises((ValueError, FileNotFoundError)) as exc_info:
        backend.load_dataset("test/invalid")
    # Either FileNotFoundError or ValueError with appropriate message
    error_message = str(exc_info.value)
    assert "Cannot load config from path" in error_message or "No such file or directory" in error_message


def test_get_dataset_type_unloaded_dataset(backend):
    """Test _get_dataset_type with unloaded dataset."""
    with pytest.raises(ValueError) as exc_info:
        backend._get_dataset_type("unloaded/dataset")
    assert "Dataset not loaded: unloaded/dataset" in str(exc_info.value)


def test_requires_3d_visualization_none_input(backend):
    """Test _requires_3d_visualization with None input."""
    # None has type 'NoneType', should return False
    result = backend._requires_3d_visualization(None)
    assert result is False


def test_requires_3d_visualization_invalid_object(backend):
    """Test _requires_3d_visualization with invalid object types."""
    # These should return False since their type names are not in REQUIRES_3D_CLASSES
    result1 = backend._requires_3d_visualization("not_a_dataset")
    assert result1 is False  # type 'str'
    
    result2 = backend._requires_3d_visualization(123)
    assert result2 is False  # type 'int'
    
    result3 = backend._requires_3d_visualization([])
    assert result3 is False  # type 'list'
    
    result4 = backend._requires_3d_visualization({})
    assert result4 is False  # type 'dict'


def test_get_dataset_type_from_inheritance_none_input(backend):
    """Test _get_dataset_type_from_inheritance with None input."""
    # None is not an instance of any dataset class, should return 'general'
    result = backend._get_dataset_type_from_inheritance(None)
    assert result == 'general'


def test_get_dataset_type_from_inheritance_invalid_object(backend):
    """Test _get_dataset_type_from_inheritance with invalid object types."""
    # These are not instances of dataset classes, should return 'general'
    result1 = backend._get_dataset_type_from_inheritance("not_a_dataset")
    assert result1 == 'general'
    
    result2 = backend._get_dataset_type_from_inheritance(123)
    assert result2 == 'general'


def test_apply_transforms_out_of_bounds_indices(backend):
    """Test _apply_transforms with out of bounds transform indices."""
    # Register one transform
    transform_dict = {
        'op': lambda x: x,  # Simple identity transform
        'input_names': [('inputs', 'data')],
        'output_names': [('inputs', 'data')]
    }
    backend._register_transform(transform_dict)
    
    # Create test datapoint
    datapoint = {
        'inputs': {'data': torch.tensor([1.0], dtype=torch.float32)},
        'labels': {'label': torch.tensor(1, dtype=torch.int64)},
        'meta_info': {'idx': 0}
    }
    
    # Test with index beyond available transforms
    with pytest.raises(IndexError):
        backend._apply_transforms(datapoint, [1], 0)
    
    with pytest.raises(IndexError):
        backend._apply_transforms(datapoint, [0, 1, 2], 0)


def test_hierarchical_datasets_invalid_config_format(backend):
    """Test get_available_datasets_hierarchical with invalid config format."""
    # Add config without proper '/' separator
    backend._configs = {
        'invalid_config_name_without_slash': {
            'path': '/fake/path',
            'type': 'semseg',
            'name': 'test'
        }
    }
    
    # Should handle gracefully or raise appropriate error
    with pytest.raises(ValueError):
        backend.get_available_datasets_hierarchical()


def test_load_dataset_corrupted_config_module(backend):
    """Test load_dataset with config that can't be executed."""
    import tempfile
    import os
    
    # Create a temporary corrupted config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("# This is a corrupted config file\n")
        f.write("invalid python syntax here !!!\n")
        f.write("data_cfg = undefined_variable\n")
        corrupted_path = f.name
    
    try:
        # Add config pointing to corrupted file
        backend._configs["test/corrupted"] = {
            'path': corrupted_path,
            'type': 'test',
            'name': 'corrupted'
        }
        
        # Should raise an error when trying to load
        with pytest.raises((NameError, SyntaxError, AttributeError)):
            backend.load_dataset("test/corrupted")
    
    finally:
        # Clean up temporary file
        if os.path.exists(corrupted_path):
            os.unlink(corrupted_path)


def test_load_dataset_config_missing_data_cfg(backend):
    """Test load_dataset with config missing data_cfg attribute."""
    import tempfile
    import os
    
    # Create a config file without data_cfg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("# Config without data_cfg\n")
        f.write("some_other_variable = 'test'\n")
        config_path = f.name
    
    try:
        # Add config pointing to file without data_cfg
        backend._configs["test/missing_data_cfg"] = {
            'path': config_path,
            'type': 'test',
            'name': 'missing_data_cfg'
        }
        
        # Should raise AttributeError when trying to access data_cfg
        with pytest.raises(AttributeError):
            backend.load_dataset("test/missing_data_cfg")
    
    finally:
        # Clean up temporary file
        if os.path.exists(config_path):
            os.unlink(config_path)


def test_load_dataset_config_invalid_data_cfg_structure(backend):
    """Test load_dataset with config having invalid data_cfg structure."""
    import tempfile
    import os
    
    # Create a config file with invalid data_cfg structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("# Config with invalid data_cfg structure\n")
        f.write("data_cfg = 'not_a_dict'\n")
        config_path = f.name
    
    try:
        # Add config pointing to file with invalid data_cfg
        backend._configs["test/invalid_structure"] = {
            'path': config_path,
            'type': 'test',
            'name': 'invalid_structure'
        }
        
        # Should raise error when trying to access train_dataset
        with pytest.raises((TypeError, KeyError)):
            backend.load_dataset("test/invalid_structure")
    
    finally:
        # Clean up temporary file
        if os.path.exists(config_path):
            os.unlink(config_path)


# Edge case tests for boundary conditions

def test_empty_dataset_handling(backend):
    """Test backend behavior with empty dataset."""
    
    class _EmptyDataset(BaseDataset):
        SPLIT_OPTIONS = ["test"]
        INPUT_NAMES = ["data"]
        LABEL_NAMES = ["label"]

        def _init_annotations(self) -> None:
            self.annotations = []  # Empty dataset

        def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
            # This should never be called for empty dataset
            raise IndexError("Dataset is empty")
        
        @staticmethod
        def display_datapoint(datapoint, class_labels=None, camera_state=None, settings_3d=None):
            """Mock display method."""
            return None
    
    empty_dataset = _EmptyDataset(split="test", device=torch.device("cpu"))
    backend._datasets["test/empty"] = empty_dataset
    
    # Getting dataset info should work
    dataset_instance = backend.get_dataset_instance("test/empty")
    assert len(dataset_instance) == 0
    
    # Getting datapoint should raise IndexError
    with pytest.raises(IndexError):
        backend.get_datapoint("test/empty", 0, [])


def test_very_large_transform_indices_list(backend, mock_dataset):
    """Test behavior with very large transform indices list."""
    # Register a few transforms
    for i in range(3):
        transform_dict = {
            'op': lambda x, i=i: x,  # Identity transform
            'input_names': [('inputs', 'data')],
            'output_names': [('inputs', 'data')]
        }
        backend._register_transform(transform_dict)
    
    backend._datasets["test/mock"] = mock_dataset
    
    # Test with very large list of valid indices
    large_indices_list = list(range(3)) * 1000  # [0,1,2,0,1,2,0,1,2,...]
    
    # Should work but might be slow
    result = backend.get_datapoint("test/mock", 0, large_indices_list)
    assert isinstance(result, dict)
    assert 'inputs' in result


def test_negative_transform_indices(backend, mock_dataset):
    """Test behavior with negative transform indices."""
    # Register transforms
    for i in range(3):
        transform_dict = {
            'op': lambda x, i=i: x,  # Identity transform
            'input_names': [('inputs', 'data')],
            'output_names': [('inputs', 'data')]
        }
        backend._register_transform(transform_dict)
    
    backend._datasets["test/mock"] = mock_dataset
    
    # Python supports negative indexing
    result = backend.get_datapoint("test/mock", 0, [-1, -2])
    assert isinstance(result, dict)
    
    # But very negative indices should fail
    with pytest.raises(IndexError):
        backend.get_datapoint("test/mock", 0, [-10])


def test_update_state_with_very_large_values(backend):
    """Test update_state with extreme values."""
    # Test with very large values
    backend.update_state(
        current_index=2**31 - 1,  # Max 32-bit int
        point_size=1e10,
        point_opacity=1e10,
        sym_diff_radius=1e10,
        corr_radius=1e10
    )
    
    # Should accept these values (no validation in current implementation)
    assert backend.current_index == 2**31 - 1
    assert backend.point_size == 1e10