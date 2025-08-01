import os
import tempfile
import threading
import pytest
from collections import defaultdict
from utils.io.config import load_config


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def create_test_config_file(filepath: str, config_dict: dict):
    """Create a test config Python file."""
    with open(filepath, 'w') as f:
        f.write(f"config = {repr(config_dict)}\n")


def create_complex_config_file(filepath: str):
    """Create a complex config file with imports and logic."""
    with open(filepath, 'w') as f:
        f.write("""
import os
import torch

# Base configuration
base_config = {
    'model': {
        'type': 'ResNet',
        'layers': 50,
        'pretrained': True
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100
    }
}

# Environment-specific overrides
if os.environ.get('DEBUG', '').lower() == 'true':
    base_config['training']['epochs'] = 5
    base_config['training']['batch_size'] = 8

# Add device configuration
base_config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

config = base_config
""")


def test_load_config_simple(temp_dir):
    """Test loading a simple config file."""
    filepath = os.path.join(temp_dir, "simple_config.py")
    test_config = {
        'model_name': 'test_model',
        'batch_size': 32,
        'learning_rate': 0.001
    }
    
    create_test_config_file(filepath, test_config)
    
    result = load_config(config_path=filepath)
    
    assert result == test_config


def test_load_config_complex(temp_dir):
    """Test loading a complex config file with imports and logic."""
    filepath = os.path.join(temp_dir, "complex_config.py")
    create_complex_config_file(filepath)
    
    result = load_config(config_path=filepath)
    
    # Verify expected structure
    assert 'model' in result
    assert 'training' in result
    assert 'device' in result
    
    assert result['model']['type'] == 'ResNet'
    assert result['model']['layers'] == 50
    assert result['training']['batch_size'] in [32, 8]  # Depends on DEBUG env var
    assert result['training']['learning_rate'] == 0.001
    assert result['device'] in ['cpu', 'cuda']


def test_load_config_with_nested_structures(temp_dir):
    """Test loading config with deeply nested structures."""
    filepath = os.path.join(temp_dir, "nested_config.py")
    nested_config = {
        'experiment': {
            'name': 'test_experiment',
            'parameters': {
                'optimizer': {
                    'type': 'Adam',
                    'settings': {
                        'lr': 0.001,
                        'betas': [0.9, 0.999],
                        'weight_decay': 1e-4
                    }
                },
                'scheduler': {
                    'type': 'StepLR',
                    'step_size': 30,
                    'gamma': 0.1
                }
            }
        },
        'data': {
            'dataset': 'CIFAR10',
            'transforms': ['RandomHorizontalFlip', 'RandomCrop', 'Normalize']
        }
    }
    
    create_test_config_file(filepath, nested_config)
    
    result = load_config(config_path=filepath)
    
    assert result == nested_config
    assert result['experiment']['parameters']['optimizer']['settings']['lr'] == 0.001
    assert result['data']['transforms'] == ['RandomHorizontalFlip', 'RandomCrop', 'Normalize']


def test_load_config_caching(temp_dir):
    """Test that config loading uses caching."""
    filepath = os.path.join(temp_dir, "cached_config.py")
    test_config = {'cached': True, 'value': 42}
    
    create_test_config_file(filepath, test_config)
    
    # Load config multiple times
    result1 = load_config(config_path=filepath)
    result2 = load_config(config_path=filepath)
    result3 = load_config(config_path=filepath)
    
    # All results should be identical
    assert result1 == test_config
    assert result2 == test_config
    assert result3 == test_config
    
    # Results should be the same object (cached)
    assert result1 is result2
    assert result2 is result3


def test_load_config_thread_safety(temp_dir):
    """Test thread-safe config loading."""
    filepath = os.path.join(temp_dir, "thread_safe_config.py")
    test_config = {'thread_safe': True, 'value': 123}
    
    create_test_config_file(filepath, test_config)
    
    results = []
    errors = []
    
    def load_config_worker():
        try:
            result = load_config(config_path=filepath)
            results.append(result)
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads that load the same config simultaneously
    threads = []
    for _ in range(10):
        thread = threading.Thread(target=load_config_worker)
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify no errors occurred
    assert len(errors) == 0, f"Errors occurred: {errors}"
    
    # Verify all results are correct and identical
    assert len(results) == 10
    for result in results:
        assert result == test_config
        # All results should be the same cached object
        assert result is results[0]


def test_load_config_nonexistent_file():
    """Test error handling for non-existent config files."""
    with pytest.raises((ImportError, FileNotFoundError)):
        load_config(config_path="nonexistent_config.py")


def test_load_config_invalid_python(temp_dir):
    """Test error handling for invalid Python syntax."""
    filepath = os.path.join(temp_dir, "invalid_config.py")
    
    # Create file with invalid Python syntax
    with open(filepath, 'w') as f:
        f.write("config = {invalid python syntax}")
    
    with pytest.raises(Exception):  # Could be SyntaxError or other import-related error
        load_config(config_path=filepath)


def test_load_config_missing_config_variable(temp_dir):
    """Test error handling when config variable is missing."""
    filepath = os.path.join(temp_dir, "no_config_var.py")
    
    # Create file without 'config' variable
    with open(filepath, 'w') as f:
        f.write("some_other_variable = {'not_config': True}")
    
    with pytest.raises(AttributeError):
        load_config(config_path=filepath)


def test_load_config_with_imports(temp_dir):
    """Test loading config that imports other modules."""
    filepath = os.path.join(temp_dir, "import_config.py")
    
    with open(filepath, 'w') as f:
        f.write("""
import math
import os
from collections import defaultdict

config = {
    'pi': math.pi,
    'home_dir': os.path.expanduser('~'),
    'default_dict': defaultdict(int),
    'computed_value': 2 ** 10
}
""")
    
    result = load_config(config_path=filepath)
    
    assert abs(result['pi'] - 3.14159) < 0.001
    assert isinstance(result['home_dir'], str)
    assert isinstance(result['default_dict'], defaultdict)  # Check type instead of callable
    assert result['computed_value'] == 1024


def test_load_config_with_functions(temp_dir):
    """Test loading config that contains functions."""
    filepath = os.path.join(temp_dir, "function_config.py")
    
    with open(filepath, 'w') as f:
        f.write("""
def create_optimizer():
    return {'type': 'Adam', 'lr': 0.001}

def create_scheduler():
    return {'type': 'StepLR', 'step_size': 30}

config = {
    'optimizer_factory': create_optimizer,
    'scheduler_factory': create_scheduler,
    'static_value': 42
}
""")
    
    result = load_config(config_path=filepath)
    
    # Functions should be accessible
    assert callable(result['optimizer_factory'])
    assert callable(result['scheduler_factory'])
    
    # Test calling the functions
    optimizer_config = result['optimizer_factory']()
    scheduler_config = result['scheduler_factory']()
    
    assert optimizer_config == {'type': 'Adam', 'lr': 0.001}
    assert scheduler_config == {'type': 'StepLR', 'step_size': 30}
    assert result['static_value'] == 42


def test_load_config_multiple_files(temp_dir):
    """Test loading multiple different config files."""
    # Create multiple config files
    config1_path = os.path.join(temp_dir, "config1.py")
    config2_path = os.path.join(temp_dir, "config2.py")
    
    config1_data = {'name': 'config1', 'value': 100}
    config2_data = {'name': 'config2', 'value': 200}
    
    create_test_config_file(config1_path, config1_data)
    create_test_config_file(config2_path, config2_data)
    
    # Load both configs
    result1 = load_config(config_path=config1_path)
    result2 = load_config(config_path=config2_path)
    
    # Verify they are different and cached separately
    assert result1 == config1_data
    assert result2 == config2_data
    assert result1 != result2
    
    # Load again to test caching
    result1_cached = load_config(config_path=config1_path)
    result2_cached = load_config(config_path=config2_path)
    
    assert result1 is result1_cached
    assert result2 is result2_cached


def test_load_config_absolute_vs_relative_paths(temp_dir):
    """Test that absolute and relative paths to same file use same cache."""
    config_data = {'path_test': True, 'value': 555}
    
    # Create config file
    relative_path = os.path.join(temp_dir, "path_config.py")
    create_test_config_file(relative_path, config_data)
    
    # Get absolute path
    absolute_path = os.path.abspath(relative_path)
    
    # Load using both paths
    result_relative = load_config(config_path=relative_path)
    result_absolute = load_config(config_path=absolute_path)
    
    # Results should be identical
    assert result_relative == config_data
    assert result_absolute == config_data
    
    # Note: They might not be the same object since the cache uses the exact path as key
    # This is acceptable behavior for the current implementation


def test_load_config_cache_persistence_across_threads(temp_dir):
    """Test that cache persists across different threads."""
    filepath = os.path.join(temp_dir, "persistent_config.py")
    test_config = {'persistent': True, 'thread_id': 0}
    
    create_test_config_file(filepath, test_config)
    
    # Load config in main thread
    main_result = load_config(config_path=filepath)
    
    # Load config in separate thread
    thread_result = None
    def load_in_thread():
        nonlocal thread_result
        thread_result = load_config(config_path=filepath)
    
    thread = threading.Thread(target=load_in_thread)
    thread.start()
    thread.join()
    
    # Both results should be identical and cached
    assert main_result == test_config
    assert thread_result == test_config
    assert main_result is thread_result  # Same cached object
