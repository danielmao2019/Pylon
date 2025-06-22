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
        self.test_data = {
            'inputs': {'img_1': np.array([1, 2, 3])},
            'labels': {},
            'meta_info': {}
        }

    def test_register_transform(self):
        """Test transform registration."""
        # Register a transform
        transform = (mock_transform1, [('inputs', 'img_1')])
        self.manager.register_transform(transform)
        self.assertEqual(len(self.manager._transforms), 1)
        self.assertEqual(self.manager._transforms[0][0], mock_transform1)
        self.assertEqual(self.manager._transforms[0][1], [('inputs', 'img_1')])

        # Register another transform
        transform2 = (mock_transform2, [('inputs', 'img_1')])
        self.manager.register_transform(transform2)
        self.assertEqual(len(self.manager._transforms), 2)

    def test_apply_transforms(self):
        """Test applying transforms."""
        # Register transforms
        transform1 = (mock_transform1, [('inputs', 'img_1')])
        transform2 = (mock_transform2, [('inputs', 'img_1')])
        self.manager.register_transform(transform1)
        self.manager.register_transform(transform2)

        # Test applying single transform
        result = self.manager.apply_transforms(self.test_data, [0])
        expected = np.array([2, 3, 4])  # Original + 1
        np.testing.assert_array_equal(result['inputs']['img_1'], expected)

        # Test applying multiple transforms
        result = self.manager.apply_transforms(self.test_data, [0, 1])
        expected = np.array([4, 6, 8])  # (Original + 1) * 2
        np.testing.assert_array_equal(result['inputs']['img_1'], expected)

    def test_clear_transforms(self):
        """Test clearing transforms."""
        transform1 = (mock_transform1, [('inputs', 'img_1')])
        transform2 = (mock_transform2, [('inputs', 'img_1')])
        self.manager.register_transform(transform1)
        self.manager.register_transform(transform2)

        self.manager.clear_transforms()
        self.assertEqual(len(self.manager._transforms), 0)
        self.assertEqual(len(self.manager.get_available_transforms()), 0)

    def test_get_transform_info(self):
        """Test getting transform info."""
        transform = (mock_transform1, [('inputs', 'img_1')])
        self.manager.register_transform(transform)

        info = self.manager.get_transform_info(0)
        self.assertEqual(info['index'], 0)
        self.assertEqual(info['name'], mock_transform1.__class__.__name__)
        self.assertEqual(info['input_keys'], [('inputs', 'img_1')])

    def test_get_available_transforms(self):
        """Test getting available transforms."""
        transform1 = (mock_transform1, [('inputs', 'img_1')])
        transform2 = (mock_transform2, [('inputs', 'img_1')])
        self.manager.register_transform(transform1)
        self.manager.register_transform(transform2)

        transforms = self.manager.get_available_transforms()
        self.assertEqual(len(transforms), 2)
        self.assertEqual(transforms[0]['name'], mock_transform1.__class__.__name__)
        self.assertEqual(transforms[1]['name'], mock_transform2.__class__.__name__)

    def test_register_transforms_from_config(self):
        """Test registering transforms from configuration."""
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
        self.manager.register_transforms_from_config(valid_config)
        self.assertEqual(len(self.manager._transforms), 2)

        # Test invalid config
        with self.assertRaises(AssertionError):
            invalid_config = {'transforms': []}
            self.manager.register_transforms_from_config(invalid_config)

        # Test non-dict config
        with self.assertRaises(AssertionError):
            self.manager.register_transforms_from_config([])  # Should raise AssertionError
