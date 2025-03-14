"""Unit tests for transform manager."""
import unittest
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

class TestTransformManager(unittest.TestCase):
    """Test cases for TransformManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = TransformManager()
        self.test_data = np.array([1, 2, 3])

    def test_register_transform(self):
        """Test transform registration."""
        # Register a transform
        self.manager.register_transform('add_one', mock_transform1)
        self.assertIn('add_one', self.manager._transforms)
        
        # Register same transform again (should overwrite)
        self.manager.register_transform('add_one', mock_transform2)
        result = self.manager.apply_transform('add_one', self.test_data)
        np.testing.assert_array_equal(result, self.test_data * 2)

    def test_get_transform(self):
        """Test getting transforms."""
        # Test getting non-existent transform
        self.assertIsNone(self.manager.get_transform('nonexistent'))
        
        # Test getting existing transform
        self.manager.register_transform('add_one', mock_transform1)
        transform = self.manager.get_transform('add_one')
        self.assertIsNotNone(transform)
        self.assertEqual(transform(1), 2)

    def test_apply_transform(self):
        """Test applying transforms."""
        self.manager.register_transform('add_one', mock_transform1)
        self.manager.register_transform('multiply_two', mock_transform2)
        
        # Test successful transform
        result = self.manager.apply_transform('add_one', self.test_data)
        np.testing.assert_array_equal(result, self.test_data + 1)
        
        # Test non-existent transform
        result = self.manager.apply_transform('nonexistent', self.test_data)
        self.assertIsNone(result)
        
        # Test failing transform
        self.manager.register_transform('fail', mock_failing_transform)
        result = self.manager.apply_transform('fail', self.test_data)
        self.assertIsNone(result)

    def test_clear_transforms(self):
        """Test clearing transforms."""
        self.manager.register_transform('add_one', mock_transform1)
        self.manager.register_transform('multiply_two', mock_transform2)
        
        self.manager.clear_transforms()
        self.assertEqual(len(self.manager._transforms), 0)
        self.assertEqual(len(self.manager.get_transform_names()), 0)

    def test_get_transform_names(self):
        """Test getting transform names."""
        self.manager.register_transform('add_one', mock_transform1)
        self.manager.register_transform('multiply_two', mock_transform2)
        
        names = self.manager.get_transform_names()
        self.assertEqual(len(names), 2)
        self.assertIn('add_one', names)
        self.assertIn('multiply_two', names)

    def test_register_transforms_from_config(self):
        """Test registering transforms from configuration."""
        # Test valid config
        valid_config = {
            'class': 'Compose',
            'args': {
                'transforms': [
                    (mock_transform1, None),
                    (mock_transform2, None)
                ]
            }
        }
        self.manager.register_transforms_from_config(valid_config)
        self.assertEqual(len(self.manager._transforms), 2)
        
        # Test invalid config
        invalid_config = {'transforms': []}
        self.manager.register_transforms_from_config(invalid_config)
        
        # Test non-dict config
        self.manager.register_transforms_from_config([])  # Should not raise exception 