import os
import json
import tempfile
import torch
import numpy as np
import pytest
from datetime import datetime
from dataclasses import dataclass
from utils.io.json import serialize_tensor, serialize_object, save_json, load_json


@dataclass
class SampleDataclass:
    """Sample dataclass for serialization tests."""
    name: str
    value: int
    tensor: torch.Tensor = None
    
    def __post_init__(self):
        if self.tensor is None:
            self.tensor = torch.tensor([1.0, 2.0, 3.0])


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_serialize_tensor_simple():
    """Test serializing simple torch tensor."""
    tensor = torch.tensor([1.0, 2.0, 3.0])
    
    result = serialize_tensor(tensor)
    
    assert isinstance(result, list)
    assert result == [1.0, 2.0, 3.0]


def test_serialize_tensor_nested_dict():
    """Test serializing nested dictionary containing tensors."""
    data = {
        'tensor1': torch.tensor([1.0, 2.0]),
        'nested': {
            'tensor2': torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
            'scalar': 42
        },
        'list': [torch.tensor([7.0]), torch.tensor([8.0])]
    }
    
    result = serialize_tensor(data)
    
    assert result['tensor1'] == [1.0, 2.0]
    assert result['nested']['tensor2'] == [[3.0, 4.0], [5.0, 6.0]]
    assert result['nested']['scalar'] == 42
    assert result['list'][0] == [7.0]
    assert result['list'][1] == [8.0]


def test_serialize_tensor_non_tensor():
    """Test that non-tensor objects pass through unchanged."""
    data = {
        'string': 'test',
        'int': 42,
        'float': 3.14,
        'list': [1, 2, 3],
        'dict': {'nested': 'value'}
    }
    
    result = serialize_tensor(data)
    
    assert result == data


def test_serialize_object_torch_tensor():
    """Test serializing torch tensors."""
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    result = serialize_object(tensor)
    
    assert isinstance(result, list)
    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_serialize_object_numpy_array():
    """Test serializing numpy arrays."""
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    result = serialize_object(array)
    
    assert isinstance(result, list)
    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_serialize_object_datetime():
    """Test serializing datetime objects."""
    dt = datetime(2023, 5, 15, 10, 30, 45)
    
    result = serialize_object(dt)
    
    assert isinstance(result, str)
    assert result == "2023-05-15T10:30:45"


def test_serialize_object_dataclass():
    """Test serializing dataclass objects."""
    obj = SampleDataclass(name="test", value=42)
    
    result = serialize_object(obj)
    
    assert isinstance(result, dict)
    assert result['name'] == "test"
    assert result['value'] == 42
    assert result['tensor'] == [1.0, 2.0, 3.0]  # Tensor should be serialized too


def test_serialize_object_complex_nested():
    """Test serializing complex nested structure."""
    data = {
        'tensor': torch.tensor([1.0, 2.0]),
        'array': np.array([3.0, 4.0]),
        'datetime': datetime(2023, 1, 1),
        'dataclass': SampleDataclass(name="nested", value=100),
        'list': [
            torch.tensor([5.0]),
            np.array([6.0]),
            datetime(2023, 2, 2)
        ],
        'tuple': (torch.tensor([7.0]), np.array([8.0])),
        'regular': {'string': 'value', 'int': 42}
    }
    
    result = serialize_object(data)
    
    assert result['tensor'] == [1.0, 2.0]
    assert result['array'] == [3.0, 4.0]
    assert result['datetime'] == "2023-01-01T00:00:00"
    assert result['dataclass']['name'] == "nested"
    assert result['dataclass']['value'] == 100
    assert result['dataclass']['tensor'] == [1.0, 2.0, 3.0]
    assert result['list'][0] == [5.0]
    assert result['list'][1] == [6.0]
    assert result['list'][2] == "2023-02-02T00:00:00"
    assert result['tuple'][0] == [7.0]
    assert result['tuple'][1] == [8.0]
    assert result['regular'] == {'string': 'value', 'int': 42}


def test_save_json_simple_data(temp_dir):
    """Test saving simple JSON data."""
    filepath = os.path.join(temp_dir, "test_save.json")
    test_data = {'key': 'value', 'number': 42, 'list': [1, 2, 3]}
    
    save_json(obj=test_data, filepath=filepath)
    
    # Verify file was created and contains correct data
    assert os.path.exists(filepath)
    with open(filepath, 'r') as f:
        loaded_data = json.load(f)
    assert loaded_data == test_data


def test_save_json_with_tensors(temp_dir):
    """Test saving data containing torch tensors."""
    filepath = os.path.join(temp_dir, "test_tensors.json")
    test_data = {
        'tensor': torch.tensor([1.0, 2.0, 3.0]),
        'nested': {
            'tensor': torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
            'regular': 'value'
        }
    }
    
    save_json(obj=test_data, filepath=filepath)
    
    # Load back and verify serialization worked
    loaded_data = load_json(filepath)
    assert loaded_data['tensor'] == [1.0, 2.0, 3.0]
    assert loaded_data['nested']['tensor'] == [[4.0, 5.0], [6.0, 7.0]]
    assert loaded_data['nested']['regular'] == 'value'


def test_save_json_with_numpy_arrays(temp_dir):
    """Test saving data containing numpy arrays."""
    filepath = os.path.join(temp_dir, "test_numpy.json")
    test_data = {
        'array': np.array([1.0, 2.0, 3.0]),
        'matrix': np.array([[4.0, 5.0], [6.0, 7.0]])
    }
    
    save_json(obj=test_data, filepath=filepath)
    
    # Load back and verify
    loaded_data = load_json(filepath)
    assert loaded_data['array'] == [1.0, 2.0, 3.0]
    assert loaded_data['matrix'] == [[4.0, 5.0], [6.0, 7.0]]


def test_save_json_with_datetime(temp_dir):
    """Test saving data containing datetime objects."""
    filepath = os.path.join(temp_dir, "test_datetime.json")
    test_data = {
        'timestamp': datetime(2023, 5, 15, 10, 30, 45),
        'created_at': datetime(2023, 1, 1, 0, 0, 0)
    }
    
    save_json(obj=test_data, filepath=filepath)
    
    # Load back and verify
    loaded_data = load_json(filepath)
    assert loaded_data['timestamp'] == "2023-05-15T10:30:45"
    assert loaded_data['created_at'] == "2023-01-01T00:00:00"


def test_save_json_with_dataclass(temp_dir):
    """Test saving data containing dataclass objects."""
    filepath = os.path.join(temp_dir, "test_dataclass.json")
    test_obj = SampleDataclass(name="test_object", value=123)
    
    save_json(obj=test_obj, filepath=filepath)
    
    # Load back and verify
    loaded_data = load_json(filepath)
    assert loaded_data['name'] == "test_object"
    assert loaded_data['value'] == 123
    assert loaded_data['tensor'] == [1.0, 2.0, 3.0]


def test_save_json_auto_create_directory(temp_dir):
    """Test that directories are automatically created."""
    nested_dir = os.path.join(temp_dir, "subdir", "nested")
    filepath = os.path.join(nested_dir, "test.json")
    test_data = {'auto_created': True}
    
    # Directory doesn't exist yet
    assert not os.path.exists(nested_dir)
    
    save_json(obj=test_data, filepath=filepath)
    
    # Directory and file should now exist
    assert os.path.exists(nested_dir)
    assert os.path.exists(filepath)
    
    # Verify content
    loaded_data = load_json(filepath)
    assert loaded_data == test_data


def test_save_json_atomic_write(temp_dir):
    """Test atomic write behavior (temp file + rename)."""
    filepath = os.path.join(temp_dir, "atomic_test.json")
    test_data = {'atomic': True, 'test': 'data'}
    
    save_json(obj=test_data, filepath=filepath)
    
    # File should exist with correct content
    assert os.path.exists(filepath)
    loaded_data = load_json(filepath)
    assert loaded_data == test_data
    
    # No temporary files should remain
    temp_files = [f for f in os.listdir(temp_dir) if f.startswith('json_') and f.endswith('.tmp')]
    assert len(temp_files) == 0


def test_save_json_pretty_formatting(temp_dir):
    """Test that saved JSON is properly formatted."""
    filepath = os.path.join(temp_dir, "formatted.json")
    test_data = {
        'key1': 'value1',
        'key2': {
            'nested': 'value',
            'number': 42
        },
        'list': [1, 2, 3]
    }
    
    save_json(obj=test_data, filepath=filepath)
    
    # Read raw file content to check formatting
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Should be formatted (multi-line, indented)
    assert '\n' in content
    assert '    ' in content  # Should have indentation


def test_save_json_overwrite_existing(temp_dir):
    """Test that save_json overwrites existing files."""
    filepath = os.path.join(temp_dir, "overwrite.json")
    
    # Save first data
    first_data = {'version': 1, 'data': 'first'}
    save_json(obj=first_data, filepath=filepath)
    
    assert os.path.exists(filepath)
    loaded_first = load_json(filepath)
    assert loaded_first == first_data
    
    # Save different data to same path
    second_data = {'version': 2, 'data': 'second', 'new_field': True}
    save_json(obj=second_data, filepath=filepath)
    
    # Verify the file was overwritten
    loaded_second = load_json(filepath)
    assert loaded_second == second_data
    assert loaded_second != first_data


def test_save_json_error_handling(temp_dir):
    """Test error handling in save_json."""
    # Test with invalid directory permissions (if possible)
    invalid_path = "/root/nonexistent/test.json"  # Likely to fail on most systems
    
    with pytest.raises(RuntimeError, match="Error saving JSON"):
        save_json(obj={'test': 'data'}, filepath=invalid_path)


def test_save_json_large_data(temp_dir):
    """Test saving large data structures."""
    filepath = os.path.join(temp_dir, "large_data.json")
    
    # Create large data structure
    large_tensor = torch.rand(100, 100)
    large_array = np.random.rand(50, 50)
    
    large_data = {
        'large_tensor': large_tensor,
        'large_array': large_array,
        'metadata': {
            'tensor_shape': list(large_tensor.shape),
            'array_shape': list(large_array.shape),
            'created_at': datetime.now()
        }
    }
    
    # Should save and load without issues
    save_json(obj=large_data, filepath=filepath)
    loaded_data = load_json(filepath)
    
    # Verify structure
    assert len(loaded_data['large_tensor']) == 100
    assert len(loaded_data['large_tensor'][0]) == 100
    assert len(loaded_data['large_array']) == 50
    assert len(loaded_data['large_array'][0]) == 50
    assert loaded_data['metadata']['tensor_shape'] == [100, 100]
    assert loaded_data['metadata']['array_shape'] == [50, 50]


def test_serialize_object_edge_cases():
    """Test serialization edge cases."""
    # Empty structures
    assert serialize_object({}) == {}
    assert serialize_object([]) == []
    assert serialize_object(()) == ()
    
    # None values
    assert serialize_object(None) is None
    assert serialize_object({'key': None}) == {'key': None}
    
    # Nested empty structures
    nested = {'empty_dict': {}, 'empty_list': [], 'empty_tuple': ()}
    assert serialize_object(nested) == nested


def test_tensor_gradient_handling():
    """Test that tensor gradients are properly detached during serialization."""
    # Create tensor with gradient
    tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = tensor.sum()
    loss.backward()
    
    # Serialize tensor with gradient
    result = serialize_tensor(tensor)
    
    # Should work without error and return the values
    assert isinstance(result, list)
    assert result == [1.0, 2.0, 3.0]


def test_save_json_concurrent_writes(temp_dir):
    """Test atomic writes prevent race conditions during concurrent saves."""
    import threading
    import time
    
    filepath = os.path.join(temp_dir, "concurrent.json")
    
    errors = []
    success_count = 0
    
    def save_worker(worker_id):
        nonlocal success_count
        try:
            data = {'worker_id': worker_id, 'timestamp': time.time()}
            save_json(obj=data, filepath=filepath)
            success_count += 1
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads that save to the same file
    threads = []
    for i in range(5):
        thread = threading.Thread(target=save_worker, args=(i,))
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify results
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert success_count == 5
    
    # File should exist and be readable (last writer wins)
    assert os.path.exists(filepath)
    final_data = load_json(filepath)
    assert 'worker_id' in final_data
    assert 'timestamp' in final_data


def test_round_trip_consistency(temp_dir):
    """Test that save/load round trip preserves data."""
    filepath = os.path.join(temp_dir, "round_trip.json")
    
    original_data = {
        'tensor': torch.tensor([1.0, 2.0, 3.0]),
        'array': np.array([[4.0, 5.0], [6.0, 7.0]]),
        'datetime': datetime(2023, 6, 15, 14, 30, 0),
        'dataclass': SampleDataclass(name="round_trip", value=999),
        'regular': {
            'string': 'test',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'null': None
        }
    }
    
    # Save and load
    save_json(obj=original_data, filepath=filepath)
    loaded_data = load_json(filepath)
    
    # Verify all serializable data is preserved
    assert loaded_data['tensor'] == [1.0, 2.0, 3.0]
    assert loaded_data['array'] == [[4.0, 5.0], [6.0, 7.0]]
    assert loaded_data['datetime'] == "2023-06-15T14:30:00"
    assert loaded_data['dataclass']['name'] == "round_trip"
    assert loaded_data['dataclass']['value'] == 999
    assert loaded_data['dataclass']['tensor'] == [1.0, 2.0, 3.0]
    assert loaded_data['regular'] == original_data['regular']
