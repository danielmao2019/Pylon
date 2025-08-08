"""Tests for ViewerBackend transform management functionality.

This module tests transform registration, clearing, application, and 
get_available_transforms functionality using real transforms and datasets.
"""

import tempfile
from typing import Dict, Any, List, Tuple

import pytest
import torch

from data.datasets.random_datasets.base_random_dataset import BaseRandomDataset
from data.transforms.base_transform import BaseTransform
from data.transforms.compose import Compose
from data.transforms.random_noise import RandomNoise
from data.transforms.identity import Identity
from data.viewer.backend.backend import ViewerBackend


@pytest.fixture
def backend():
    """Create a ViewerBackend instance for testing."""
    return ViewerBackend()


@pytest.fixture
def random_dataset():
    """Create a random dataset for testing transforms."""
    # Simple generator configuration for images and labels
    gen_func_config = {
        'inputs': {
            'image': (lambda **kwargs: torch.randn(3, 224, 224, dtype=torch.float32), {})
        },
        'labels': {
            'label': (lambda **kwargs: torch.randint(0, 10, (1,), dtype=torch.int64), {})
        }
    }
    
    dataset = BaseRandomDataset(
        num_examples=5,
        gen_func_config=gen_func_config,
        split="all",
        device=torch.device("cpu")
    )
    return dataset


def test_clear_transforms_initial_state(backend):
    """Test that _clear_transforms works on empty transform list."""
    # Initially transforms should be empty
    assert len(backend._transforms) == 0
    
    # Clearing empty transforms should work
    backend._clear_transforms()
    assert len(backend._transforms) == 0


def test_clear_transforms_with_existing_transforms(backend):
    """Test that _clear_transforms removes all transforms."""
    # Add some real transforms directly
    transforms = [
        {
            'op': Identity(),
            'input_names': [('inputs', 'image')],
            'output_names': [('inputs', 'image')]
        },
        {
            'op': RandomNoise(noise_std=0.1),
            'input_names': [('inputs', 'image')],
            'output_names': [('inputs', 'image')]
        }
    ]
    
    for transform in transforms:
        backend._register_transform(transform)
    
    # Verify transforms were added
    assert len(backend._transforms) == 2
    
    # Clear transforms
    backend._clear_transforms()
    
    # Verify all transforms were removed
    assert len(backend._transforms) == 0


def test_register_transform_single_transform(backend):
    """Test registering a single transform."""
    transform_dict = {
        'op': Identity(),
        'input_names': [('inputs', 'image')],
        'output_names': [('inputs', 'image')]
    }
    
    backend._register_transform(transform_dict)
    
    # Verify transform was registered
    assert len(backend._transforms) == 1
    assert backend._transforms[0] == transform_dict


def test_register_transform_multiple_transforms(backend):
    """Test registering multiple transforms."""
    transforms = [
        {
            'op': Identity(),
            'input_names': [('inputs', 'image')],
            'output_names': [('inputs', 'image')]
        },
        {
            'op': RandomNoise(degrees=30),
            'input_names': [('inputs', 'image')],
            'output_names': [('inputs', 'image')]
        }
    ]
    
    for transform in transforms:
        backend._register_transform(transform)
    
    # Verify all transforms were registered in order
    assert len(backend._transforms) == 2
    for i, expected_transform in enumerate(transforms):
        assert backend._transforms[i] == expected_transform


def test_get_available_transforms_empty_list(backend):
    """Test get_available_transforms with no transforms registered."""
    transforms_info = backend.get_available_transforms()
    
    assert isinstance(transforms_info, list)
    assert len(transforms_info) == 0


def test_get_available_transforms_with_real_transforms(backend):
    """Test get_available_transforms with real transform classes."""
    transforms = [
        {
            'op': Identity(),
            'input_names': [('inputs', 'image')],
            'output_names': [('inputs', 'image')]
        },
        {
            'op': RandomNoise(noise_std=0.1),
            'input_names': [('inputs', 'image')],
            'output_names': [('inputs', 'image')]
        }
    ]
    
    for transform in transforms:
        backend._register_transform(transform)
    
    transforms_info = backend.get_available_transforms()
    
    # Verify structure
    assert len(transforms_info) == 2
    
    # Verify first transform info
    transform_info_0 = transforms_info[0]
    assert transform_info_0['index'] == 0
    assert transform_info_0['name'] == 'Identity'
    assert 'Identity' in transform_info_0['string']
    assert transform_info_0['input_names'] == [('inputs', 'image')]
    assert transform_info_0['output_names'] == [('inputs', 'image')]
    
    # Verify second transform info
    transform_info_1 = transforms_info[1]
    assert transform_info_1['index'] == 1
    assert transform_info_1['name'] == 'RandomNoise'
    assert 'RandomNoise' in transform_info_1['string']
    assert transform_info_1['input_names'] == [('inputs', 'image')]
    assert transform_info_1['output_names'] == [('inputs', 'image')]


def test_get_available_transforms_config_based_transforms(backend):
    """Test get_available_transforms with config-based transform definitions."""
    # Test with config-style transform definition
    transform_dict = {
        'op': {
            'class': RandomNoise,
            'args': {'noise_std': 0.3}
        },
        'input_names': [('inputs', 'image')],
        'output_names': [('inputs', 'image')]
    }
    
    backend._register_transform(transform_dict)
    transforms_info = backend.get_available_transforms()
    
    assert len(transforms_info) == 1
    transform_info = transforms_info[0]
    
    assert transform_info['name'] == 'RandomNoise'
    assert 'RandomNoise' in transform_info['string']
    assert 'noise_std=0.3' in transform_info['string']  # Parameters should be formatted


def test_apply_transforms_empty_list(backend):
    """Test _apply_transforms with empty transform list."""
    # Create test datapoint
    datapoint = {
        'inputs': {'image': torch.randn(3, 224, 224, dtype=torch.float32)},
        'labels': {'label': torch.tensor(1, dtype=torch.int64)},
        'meta_info': {'idx': 0}
    }
    
    # Apply empty transform list
    result = backend._apply_transforms(datapoint, [], 0)
    
    # Should return original datapoint unchanged
    assert result == datapoint
    
    # Verify tensors are the same objects (no copying)
    assert result['inputs']['image'] is datapoint['inputs']['image']
    assert result['labels']['label'] is datapoint['labels']['label']


def test_apply_transforms_with_identity_transform(backend):
    """Test _apply_transforms with identity transform."""
    # Create a simple identity transform
    class IdentityTransform(BaseTransform):
        def _call_single(self, inputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
            return inputs
    
    # Register transform
    transform_dict = {
        'op': IdentityTransform(),
        'input_names': [('inputs', 'image')],
        'output_names': [('inputs', 'image')]
    }
    backend._register_transform(transform_dict)
    
    # Create test datapoint
    original_image = torch.randn(3, 224, 224, dtype=torch.float32)
    datapoint = {
        'inputs': {'image': original_image.clone()},
        'labels': {'label': torch.tensor(1, dtype=torch.int64)},
        'meta_info': {'idx': 0}
    }
    
    # Apply transform
    result = backend._apply_transforms(datapoint, [0], 0)
    
    # Should return data unchanged (identity)
    torch.testing.assert_close(result['inputs']['image'], original_image)


def test_transform_workflow_integration_with_dataset(backend, random_dataset):
    """Test complete transform workflow integration with real dataset."""
    # Create real transforms for image data
    transforms = [
        (Identity(), [('inputs', 'image')]),
        (RandomNoise(noise_std=0.0), [('inputs', 'image')])  # 0 std = no noise
    ]
    
    # Create Compose with transforms
    compose = Compose(transforms=transforms)
    
    # Set dataset transforms
    random_dataset.transforms = compose
    
    # Simulate the dataset loading process (like load_dataset would do)
    backend._clear_transforms()
    for transform in random_dataset.transforms.transforms:
        backend._register_transform(transform)
    
    # Clear transforms from dataset (like load_dataset does)
    random_dataset.transforms = Compose(transforms=[])
    
    # Get available transforms
    available_transforms = backend.get_available_transforms()
    assert len(available_transforms) == 2
    
    # Get datapoint without transforms
    datapoint = random_dataset[0]
    
    # The transforms we registered should be available
    assert len(backend._transforms) == 2


def test_apply_transforms_deterministic_seeding(backend):
    """Test that _apply_transforms produces deterministic results with seeding."""
    # Use RandomNoise to test deterministic seeding
    transform_dict = {
        'op': RandomNoise(noise_std=0.1),
        'input_names': [('inputs', 'image')],
        'output_names': [('inputs', 'image')]
    }
    backend._register_transform(transform_dict)
    
    # Create test datapoint
    original_image = torch.randn(3, 224, 224, dtype=torch.float32)
    
    # Apply transform multiple times with same seed
    results = []
    for _ in range(3):
        datapoint = {
            'inputs': {'image': original_image.clone()},
            'labels': {'label': torch.tensor(1, dtype=torch.int64)},
            'meta_info': {'idx': 0}
        }
        result = backend._apply_transforms(datapoint, [0], 42)  # Same seed
        results.append(result['inputs']['image'].clone())
    
    # All results should be identical (deterministic with same seed)
    for i in range(1, len(results)):
        torch.testing.assert_close(results[0], results[i])


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_apply_transforms_invalid_transform_index(backend):
    """Test _apply_transforms with invalid transform index."""
    # Register one transform
    transform_dict = {
        'op': Identity(),
        'input_names': [('inputs', 'image')],
        'output_names': [('inputs', 'image')]
    }
    backend._register_transform(transform_dict)
    
    # Create test datapoint
    datapoint = {
        'inputs': {'image': torch.randn(3, 224, 224, dtype=torch.float32)},
        'labels': {'label': torch.tensor(1, dtype=torch.int64)},
        'meta_info': {'idx': 0}
    }
    
    # Try to apply transform with invalid index
    with pytest.raises(IndexError):
        backend._apply_transforms(datapoint, [1], 0)  # Index 1 doesn't exist
    
    with pytest.raises(IndexError):
        backend._apply_transforms(datapoint, [5], 0)  # Index 5 doesn't exist


def test_apply_transforms_out_of_bounds_indices(backend):
    """Test _apply_transforms with out of bounds transform indices."""
    # Register one transform
    transform_dict = {
        'op': Identity(),
        'input_names': [('inputs', 'image')],
        'output_names': [('inputs', 'image')]
    }
    backend._register_transform(transform_dict)
    
    # Create test datapoint
    datapoint = {
        'inputs': {'image': torch.randn(3, 224, 224, dtype=torch.float32)},
        'labels': {'label': torch.tensor(1, dtype=torch.int64)},
        'meta_info': {'idx': 0}
    }
    
    # Test with index beyond available transforms
    with pytest.raises(IndexError):
        backend._apply_transforms(datapoint, [1], 0)
    
    with pytest.raises(IndexError):
        backend._apply_transforms(datapoint, [0, 1, 2], 0)


def test_apply_transforms_negative_index(backend):
    """Test _apply_transforms with negative transform index."""
    # Register one transform
    transform_dict = {
        'op': Identity(),
        'input_names': [('inputs', 'image')],
        'output_names': [('inputs', 'image')]
    }
    backend._register_transform(transform_dict)
    
    # Create test datapoint
    datapoint = {
        'inputs': {'image': torch.randn(3, 224, 224, dtype=torch.float32)},
        'labels': {'label': torch.tensor(1, dtype=torch.int64)},
        'meta_info': {'idx': 0}
    }
    
    # Negative indices should work (Python list indexing)
    result = backend._apply_transforms(datapoint, [-1], 0)
    assert isinstance(result, dict)
    assert 'inputs' in result
    
    # But very negative indices should fail
    with pytest.raises(IndexError):
        backend._apply_transforms(datapoint, [-10], 0)
