import os
import json
import tempfile
import torch
import numpy as np
import pytest
from datetime import datetime
from dataclasses import dataclass
from utils.io.json import load_json


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_load_json_valid_file(temp_dir):
    """Test loading valid JSON file."""
    filepath = os.path.join(temp_dir, "test.json")
    test_data = {'key': 'value', 'number': 42, 'list': [1, 2, 3]}

    # Create test file
    with open(filepath, 'w') as f:
        json.dump(test_data, f)

    result = load_json(filepath)

    assert result == test_data


def test_load_json_nested_structure(temp_dir):
    """Test loading JSON with nested structures."""
    filepath = os.path.join(temp_dir, "nested.json")
    test_data = {
        'level1': {
            'level2': {'level3': {'deep_value': 'found', 'numbers': [1, 2, 3, 4, 5]}},
            'simple': 'value',
        },
        'top_level': 'data',
    }

    # Create test file
    with open(filepath, 'w') as f:
        json.dump(test_data, f)

    result = load_json(filepath)

    assert result == test_data
    assert result['level1']['level2']['level3']['deep_value'] == 'found'
    assert result['level1']['level2']['level3']['numbers'] == [1, 2, 3, 4, 5]


def test_load_json_various_data_types(temp_dir):
    """Test loading JSON with various Python data types."""
    filepath = os.path.join(temp_dir, "types.json")
    test_data = {
        'string': 'hello world',
        'integer': 42,
        'float': 3.14159,
        'boolean_true': True,
        'boolean_false': False,
        'null_value': None,
        'list': [1, 'two', 3.0, True, None],
        'empty_dict': {},
        'empty_list': [],
    }

    # Create test file
    with open(filepath, 'w') as f:
        json.dump(test_data, f)

    result = load_json(filepath)

    assert result == test_data
    assert isinstance(result['string'], str)
    assert isinstance(result['integer'], int)
    assert isinstance(result['float'], float)
    assert isinstance(result['boolean_true'], bool)
    assert isinstance(result['boolean_false'], bool)
    assert result['null_value'] is None
    assert isinstance(result['list'], list)
    assert isinstance(result['empty_dict'], dict)
    assert isinstance(result['empty_list'], list)


def test_load_json_unicode_content(temp_dir):
    """Test loading JSON with unicode characters."""
    filepath = os.path.join(temp_dir, "unicode.json")
    test_data = {
        'english': 'Hello World',
        'chinese': '你好世界',
        'japanese': 'こんにちは世界',
        'emoji': '🌍🚀💻',
        'special_chars': 'åäöñüß',
        'symbols': '∑∞∆∇',
    }

    # Create test file with UTF-8 encoding
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)

    result = load_json(filepath)

    assert result == test_data
    assert result['chinese'] == '你好世界'
    assert result['emoji'] == '🌍🚀💻'


def test_load_json_large_data(temp_dir):
    """Test loading reasonably large JSON data."""
    filepath = os.path.join(temp_dir, "large.json")

    # Create large data structure
    test_data = {
        'arrays': {
            f'array_{i}': list(range(100 * i, 100 * (i + 1))) for i in range(10)
        },
        'metadata': {'total_items': 1000, 'description': 'Large test dataset'},
    }

    # Create test file
    with open(filepath, 'w') as f:
        json.dump(test_data, f)

    result = load_json(filepath)

    assert result == test_data
    assert len(result['arrays']) == 10
    assert len(result['arrays']['array_0']) == 100
    assert result['arrays']['array_0'][0] == 0
    assert result['arrays']['array_9'][-1] == 999


def test_load_json_previously_serialized_data(temp_dir):
    """Test loading JSON that was previously serialized from complex objects."""
    filepath = os.path.join(temp_dir, "serialized.json")

    # Simulate data that was serialized from tensors, arrays, etc.
    test_data = {
        'tensor_data': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # Serialized 2D tensor
        'array_data': [1.0, 2.0, 3.0, 4.0, 5.0],  # Serialized 1D array
        'datetime_string': '2023-05-15T10:30:45',  # Serialized datetime
        'dataclass_data': {  # Serialized dataclass
            'name': 'test_object',
            'value': 123,
            'tensor': [1.0, 2.0, 3.0],
        },
    }

    # Create test file
    with open(filepath, 'w') as f:
        json.dump(test_data, f)

    result = load_json(filepath)

    assert result == test_data
    assert result['tensor_data'] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert result['datetime_string'] == '2023-05-15T10:30:45'
    assert result['dataclass_data']['name'] == 'test_object'


def test_load_json_formatted_file(temp_dir):
    """Test loading JSON file with pretty formatting."""
    filepath = os.path.join(temp_dir, "formatted.json")
    test_data = {
        'key1': 'value1',
        'key2': {'nested': 'value', 'number': 42},
        'list': [1, 2, 3],
    }

    # Create formatted JSON file
    with open(filepath, 'w') as f:
        json.dump(test_data, f, indent=4, sort_keys=True)

    result = load_json(filepath)

    assert result == test_data


def test_load_json_minified_file(temp_dir):
    """Test loading minified JSON file."""
    filepath = os.path.join(temp_dir, "minified.json")
    test_data = {'a': 1, 'b': [2, 3, 4], 'c': {'d': 'e'}}

    # Create minified JSON file
    with open(filepath, 'w') as f:
        json.dump(test_data, f, separators=(',', ':'))

    result = load_json(filepath)

    assert result == test_data


def test_load_json_nonexistent_file():
    """Test error handling for non-existent file."""
    with pytest.raises(RuntimeError, match="File does not exist"):
        load_json("nonexistent.json")


def test_load_json_empty_file(temp_dir):
    """Test error handling for empty file."""
    filepath = os.path.join(temp_dir, "empty.json")

    # Create empty file
    open(filepath, 'w').close()

    with pytest.raises(RuntimeError, match="File is empty"):
        load_json(filepath)


def test_load_json_invalid_json(temp_dir):
    """Test error handling for invalid JSON."""
    filepath = os.path.join(temp_dir, "invalid.json")

    # Create file with invalid JSON
    with open(filepath, 'w') as f:
        f.write("invalid json content {")

    with pytest.raises(RuntimeError, match="Error loading JSON"):
        load_json(filepath)


def test_load_json_malformed_syntax(temp_dir):
    """Test error handling for various malformed JSON syntax."""
    malformed_cases = [
        ("unclosed_brace", '{"key": "value"'),
        ("trailing_comma", '{"key": "value",}'),
        ("unquoted_keys", '{key: "value"}'),
        ("single_quotes", "{'key': 'value'}"),
        ("undefined_value", '{"key": undefined}'),
    ]

    for name, content in malformed_cases:
        filepath = os.path.join(temp_dir, f"malformed_{name}.json")

        with open(filepath, 'w') as f:
            f.write(content)

        with pytest.raises(RuntimeError, match="Error loading JSON"):
            load_json(filepath)


def test_load_json_permission_error():
    """Test error handling when file exists but can't be read due to permissions."""
    # This test might not work on all systems, so we'll skip it if permission changes fail
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_filepath = f.name

        # Try to make file unreadable (might not work on all systems)
        try:
            os.chmod(temp_filepath, 0o000)

            with pytest.raises(RuntimeError):
                load_json(temp_filepath)

        except (OSError, PermissionError):
            # If we can't change permissions, skip this test
            pytest.skip("Cannot test permission errors on this system")
        finally:
            # Restore permissions and clean up
            try:
                os.chmod(temp_filepath, 0o644)
                os.unlink(temp_filepath)
            except (OSError, PermissionError):
                pass
    except Exception:
        pytest.skip("Permission test setup failed")


def test_load_json_concurrent_access(temp_dir):
    """Test loading JSON file that might be accessed concurrently."""
    import threading
    import time

    filepath = os.path.join(temp_dir, "concurrent.json")
    test_data = {'concurrent': True, 'data': list(range(100))}

    # Create test file
    with open(filepath, 'w') as f:
        json.dump(test_data, f)

    results = []
    errors = []

    def load_worker():
        try:
            result = load_json(filepath)
            results.append(result)
        except Exception as e:
            errors.append(e)

    # Create multiple threads that load the same file
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=load_worker)
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify results
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 5
    for result in results:
        assert result == test_data


def test_load_json_special_float_values(temp_dir):
    """Test loading JSON with special float values that Python can handle."""
    filepath = os.path.join(temp_dir, "special_floats.json")

    # Note: JSON doesn't support NaN, Infinity, etc., but we can test very large/small numbers
    test_data = {
        'very_large': 1e100,
        'very_small': 1e-100,
        'zero': 0.0,
        'negative_zero': -0.0,
        'large_negative': -1e50,
    }

    # Create test file
    with open(filepath, 'w') as f:
        json.dump(test_data, f)

    result = load_json(filepath)

    assert result == test_data
    assert result['very_large'] == 1e100
    assert result['very_small'] == 1e-100
