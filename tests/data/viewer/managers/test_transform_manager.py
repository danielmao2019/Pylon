"""Unit tests for transform manager."""
import pytest
import numpy as np
from data.viewer.managers.transform_manager import TransformManager


def mock_transform1(data):
    """Mock transform that adds 1 to all elements."""
    return data + 1


def mock_transform2(data):
    """Mock transform that multiplies all elements by 2."""
    return data * 2


def mock_failing_transform(data):
    """Mock transform that raises an exception."""
    raise ValueError("Mock transform error")


def get_test_manager():
    """Get a test transform manager instance."""
    return TransformManager()


def get_test_data():
    """Get test data."""
    return {
        'inputs': {'img_1': np.array([1, 2, 3])},
        'labels': {},
        'meta_info': {}
    }

def test_register_transform():
    """Test transform registration."""
    manager = get_test_manager()
    
    # Register a transform
    transform = (mock_transform1, [('inputs', 'img_1')])
    manager.register_transform(transform)
    assert len(manager._transforms) == 1
    assert manager._transforms[0][0] == mock_transform1
    assert manager._transforms[0][1] == [('inputs', 'img_1')]

    # Register another transform
    transform2 = (mock_transform2, [('inputs', 'img_1')])
    manager.register_transform(transform2)
    assert len(manager._transforms) == 2

def test_apply_transforms():
    """Test applying transforms."""
    manager = get_test_manager()
    test_data = get_test_data()
    
    # Register transforms
    transform1 = (mock_transform1, [('inputs', 'img_1')])
    transform2 = (mock_transform2, [('inputs', 'img_1')])
    manager.register_transform(transform1)
    manager.register_transform(transform2)

    # Test applying single transform
    result = manager.apply_transforms(test_data, [0])
    expected = np.array([2, 3, 4])  # Original + 1
    np.testing.assert_array_equal(result['inputs']['img_1'], expected)

    # Test applying multiple transforms
    result = manager.apply_transforms(test_data, [0, 1])
    expected = np.array([4, 6, 8])  # (Original + 1) * 2
    np.testing.assert_array_equal(result['inputs']['img_1'], expected)

def test_clear_transforms():
    """Test clearing transforms."""
    manager = get_test_manager()
    
    transform1 = (mock_transform1, [('inputs', 'img_1')])
    transform2 = (mock_transform2, [('inputs', 'img_1')])
    manager.register_transform(transform1)
    manager.register_transform(transform2)

    manager.clear_transforms()
    assert len(manager._transforms) == 0
    assert len(manager.get_available_transforms()) == 0

def test_get_transform_info():
    """Test getting transform info."""
    manager = get_test_manager()
    
    transform = (mock_transform1, [('inputs', 'img_1')])
    manager.register_transform(transform)

    info = manager.get_transform_info(0)
    assert info['index'] == 0
    assert info['name'] == mock_transform1.__class__.__name__
    assert info['input_keys'] == [('inputs', 'img_1')]

def test_get_available_transforms():
    """Test getting available transforms."""
    manager = get_test_manager()
    
    transform1 = (mock_transform1, [('inputs', 'img_1')])
    transform2 = (mock_transform2, [('inputs', 'img_1')])
    manager.register_transform(transform1)
    manager.register_transform(transform2)

    transforms = manager.get_available_transforms()
    assert len(transforms) == 2
    assert transforms[0]['name'] == mock_transform1.__class__.__name__
    assert transforms[1]['name'] == mock_transform2.__class__.__name__

def test_register_transforms_from_config():
    """Test registering transforms from configuration."""
    manager = get_test_manager()
    
    # Test valid config
    valid_config = {
        'class': 'Compose',
        'args': {
            'transforms': [
                (mock_transform1, [('inputs', 'img_1')]),
                (mock_transform2, [('inputs', 'img_1')])
            ]
        }
    }
    manager.register_transforms_from_config(valid_config)
    assert len(manager._transforms) == 2

    # Test invalid config
    with pytest.raises(AssertionError):
        invalid_config = {'transforms': []}
        manager.register_transforms_from_config(invalid_config)

    # Test non-dict config
    with pytest.raises(AssertionError):
        manager.register_transforms_from_config([])  # Should raise AssertionError
