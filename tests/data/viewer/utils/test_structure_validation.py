"""Tests for datapoint structure validation utilities.

This module tests the structure validation functions that ensure datapoints
match the expected format for each dataset type before display.
"""
import pytest
import torch
from typing import Dict, Any
from data.viewer.utils.structure_validation import (
    _validate_basic_datapoint_structure,
    validate_semseg_structure,
    validate_2dcd_structure,
    validate_3dcd_structure,
    validate_pcr_structure,
    validate_structure_for_type
)


# Fixtures for valid datapoints


@pytest.fixture
def valid_semseg_datapoint():
    """Valid semantic segmentation datapoint."""
    return {
        'inputs': {
            'image': torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        },
        'labels': {
            'label': torch.randint(0, 10, (64, 64), dtype=torch.long)
        },
        'meta_info': {
            'image_path': '/path/to/image.jpg',
            'index': 0
        }
    }


@pytest.fixture
def valid_semseg_instance_datapoint():
    """Valid semantic segmentation datapoint with instance format."""
    return {
        'inputs': {
            'image': torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        },
        'labels': {
            'label': {
                'masks': [torch.randint(0, 2, (64, 64), dtype=torch.bool) for _ in range(3)],
                'indices': [1, 2, 3]
            }
        },
        'meta_info': {
            'image_path': '/path/to/image.jpg',
            'index': 0
        }
    }


@pytest.fixture
def valid_2dcd_datapoint():
    """Valid 2D change detection datapoint."""
    return {
        'inputs': {
            'img_1': torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8),
            'img_2': torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        },
        'labels': {
            'change_map': torch.randint(0, 2, (64, 64), dtype=torch.long)
        },
        'meta_info': {
            'img1_path': '/path/to/img1.jpg',
            'img2_path': '/path/to/img2.jpg',
            'index': 0
        }
    }


@pytest.fixture
def valid_3dcd_datapoint():
    """Valid 3D change detection datapoint."""
    n_points = 1000
    return {
        'inputs': {
            'pc_1': {
                'pos': torch.randn(n_points, 3, dtype=torch.float32),
                'features': torch.randn(n_points, 64, dtype=torch.float32)
            },
            'pc_2': {
                'pos': torch.randn(n_points, 3, dtype=torch.float32),
                'features': torch.randn(n_points, 64, dtype=torch.float32)
            }
        },
        'labels': {
            'change_map': torch.randint(0, 2, (n_points,), dtype=torch.long)
        },
        'meta_info': {
            'pc1_path': '/path/to/pc1.pcd',
            'pc2_path': '/path/to/pc2.pcd',
            'index': 0
        }
    }


@pytest.fixture
def valid_pcr_datapoint():
    """Valid point cloud registration datapoint."""
    n_points = 1000
    return {
        'inputs': {
            'src_pc': {
                'pos': torch.randn(n_points, 3, dtype=torch.float32),
                'features': torch.randn(n_points, 64, dtype=torch.float32)
            },
            'tgt_pc': {
                'pos': torch.randn(n_points, 3, dtype=torch.float32),
                'features': torch.randn(n_points, 64, dtype=torch.float32)
            },
            'correspondences': torch.randint(0, n_points, (500, 2), dtype=torch.long)
        },
        'labels': {
            'transform': torch.eye(4, dtype=torch.float32)
        },
        'meta_info': {
            'src_path': '/path/to/src.pcd',
            'tgt_path': '/path/to/tgt.pcd',
            'index': 0
        }
    }


@pytest.fixture
def valid_pcr_batched_datapoint():
    """Valid batched point cloud registration datapoint (from collators)."""
    n_points_total = 2000
    return {
        'inputs': {
            'points': torch.randn(n_points_total, 3, dtype=torch.float32),
            'lengths': torch.tensor([1000, 1000], dtype=torch.long),
            'features': torch.randn(n_points_total, 64, dtype=torch.float32)
        },
        'labels': {
            'transform': torch.eye(4, dtype=torch.float32)  # Should be 2D [4,4], not 3D
        },
        'meta_info': {
            'batch_size': 2,
            'index': 0
        }
    }


# Basic structure validation tests


def test_validate_basic_datapoint_structure_valid(valid_semseg_datapoint):
    """Test basic structure validation with valid datapoint."""
    # Should not raise any exception
    _validate_basic_datapoint_structure(valid_semseg_datapoint)


def test_validate_basic_datapoint_structure_invalid_type():
    """Test basic structure validation with invalid datapoint type."""
    with pytest.raises(AssertionError) as exc_info:
        _validate_basic_datapoint_structure("not_a_dict")
    
    assert "Datapoint must be a dictionary" in str(exc_info.value)


def test_validate_basic_datapoint_structure_missing_keys():
    """Test basic structure validation with missing required keys."""
    # Missing 'labels' key
    invalid_datapoint = {
        'inputs': {'image': torch.randn(3, 32, 32)},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        _validate_basic_datapoint_structure(invalid_datapoint)
    
    assert "Missing required top-level keys" in str(exc_info.value)
    assert "labels" in str(exc_info.value)


def test_validate_basic_datapoint_structure_empty_inputs():
    """Test basic structure validation with empty inputs."""
    invalid_datapoint = {
        'inputs': {},  # Empty inputs
        'labels': {'label': torch.tensor(1)},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        _validate_basic_datapoint_structure(invalid_datapoint)
    
    assert "'inputs' dictionary must not be empty" in str(exc_info.value)


def test_validate_basic_datapoint_structure_empty_labels():
    """Test basic structure validation with empty labels."""
    invalid_datapoint = {
        'inputs': {'image': torch.randn(3, 32, 32)},
        'labels': {},  # Empty labels
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        _validate_basic_datapoint_structure(invalid_datapoint)
    
    assert "'labels' dictionary must not be empty" in str(exc_info.value)


def test_validate_basic_datapoint_structure_invalid_inputs_type():
    """Test basic structure validation with invalid inputs type."""
    invalid_datapoint = {
        'inputs': "not_a_dict",  # Should be dict
        'labels': {'label': torch.tensor(1)},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        _validate_basic_datapoint_structure(invalid_datapoint)
    
    assert "'inputs' must be a dictionary" in str(exc_info.value)


# Semantic segmentation validation tests


def test_validate_semseg_structure_valid_tensor(valid_semseg_datapoint):
    """Test semantic segmentation validation with valid tensor label."""
    # Should not raise any exception
    validate_semseg_structure(valid_semseg_datapoint)


def test_validate_semseg_structure_valid_instance(valid_semseg_instance_datapoint):
    """Test semantic segmentation validation with valid instance format."""
    # Should not raise any exception
    validate_semseg_structure(valid_semseg_instance_datapoint)


def test_validate_semseg_structure_missing_image():
    """Test semantic segmentation validation with missing image key."""
    invalid_datapoint = {
        'inputs': {'not_image': torch.randn(3, 32, 32)},  # Wrong key
        'labels': {'label': torch.randint(0, 10, (32, 32))},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_semseg_structure(invalid_datapoint)
    
    assert "inputs must have 'image' key" in str(exc_info.value)


def test_validate_semseg_structure_invalid_image_dimensions():
    """Test semantic segmentation validation with invalid image dimensions."""
    invalid_datapoint = {
        'inputs': {'image': torch.randn(32, 32)},  # Should be 3D
        'labels': {'label': torch.randint(0, 10, (32, 32))},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_semseg_structure(invalid_datapoint)
    
    assert "inputs['image'] must be 3D tensor [C,H,W]" in str(exc_info.value)


def test_validate_semseg_structure_missing_label():
    """Test semantic segmentation validation with missing label key."""
    invalid_datapoint = {
        'inputs': {'image': torch.randn(3, 32, 32)},
        'labels': {'not_label': torch.randint(0, 10, (32, 32))},  # Wrong key
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_semseg_structure(invalid_datapoint)
    
    assert "labels must have 'label' key" in str(exc_info.value)


def test_validate_semseg_structure_invalid_label_type():
    """Test semantic segmentation validation with invalid label type."""
    invalid_datapoint = {
        'inputs': {'image': torch.randn(3, 32, 32)},
        'labels': {'label': "not_tensor_or_dict"},  # Should be tensor or dict
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_semseg_structure(invalid_datapoint)
    
    assert "labels['label'] must be torch.Tensor or dict" in str(exc_info.value)


def test_validate_semseg_structure_invalid_instance_format():
    """Test semantic segmentation validation with invalid instance format."""
    invalid_datapoint = {
        'inputs': {'image': torch.randn(3, 32, 32)},
        'labels': {'label': {'masks': []}},  # Missing 'indices' key
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_semseg_structure(invalid_datapoint)
    
    assert "labels['label'] dict must have 'masks' and 'indices' keys" in str(exc_info.value)


# 2D change detection validation tests


def test_validate_2dcd_structure_valid(valid_2dcd_datapoint):
    """Test 2D change detection validation with valid datapoint."""
    # Should not raise any exception
    validate_2dcd_structure(valid_2dcd_datapoint)


def test_validate_2dcd_structure_missing_image():
    """Test 2D change detection validation with missing image key."""
    invalid_datapoint = {
        'inputs': {'img_1': torch.randn(3, 32, 32)},  # Missing img_2
        'labels': {'change_map': torch.randint(0, 2, (32, 32))},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_2dcd_structure(invalid_datapoint)
    
    assert "inputs must have 'img_2' key" in str(exc_info.value)


def test_validate_2dcd_structure_shape_mismatch():
    """Test 2D change detection validation with image shape mismatch."""
    invalid_datapoint = {
        'inputs': {
            'img_1': torch.randn(3, 32, 32),
            'img_2': torch.randn(3, 64, 64)  # Different shape
        },
        'labels': {'change_map': torch.randint(0, 2, (32, 32))},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_2dcd_structure(invalid_datapoint)
    
    assert "Image shapes don't match" in str(exc_info.value)


def test_validate_2dcd_structure_missing_change_map():
    """Test 2D change detection validation with missing change map."""
    invalid_datapoint = {
        'inputs': {
            'img_1': torch.randn(3, 32, 32),
            'img_2': torch.randn(3, 32, 32)
        },
        'labels': {'not_change_map': torch.randint(0, 2, (32, 32))},  # Wrong key
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_2dcd_structure(invalid_datapoint)
    
    assert "labels must have 'change_map' key" in str(exc_info.value)


def test_validate_2dcd_structure_invalid_change_map_dimensions():
    """Test 2D change detection validation with invalid change map dimensions."""
    invalid_datapoint = {
        'inputs': {
            'img_1': torch.randn(3, 32, 32),
            'img_2': torch.randn(3, 32, 32)
        },
        'labels': {'change_map': torch.randint(0, 2, (32,))},  # Should be at least 2D
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_2dcd_structure(invalid_datapoint)
    
    assert "labels['change_map'] must be at least 2D [H,W]" in str(exc_info.value)


# 3D change detection validation tests


def test_validate_3dcd_structure_valid(valid_3dcd_datapoint):
    """Test 3D change detection validation with valid datapoint."""
    # Should not raise any exception
    validate_3dcd_structure(valid_3dcd_datapoint)


def test_validate_3dcd_structure_missing_pc():
    """Test 3D change detection validation with missing point cloud."""
    invalid_datapoint = {
        'inputs': {'pc_1': {'pos': torch.randn(100, 3)}},  # Missing pc_2
        'labels': {'change_map': torch.randint(0, 2, (100,))},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_3dcd_structure(invalid_datapoint)
    
    assert "inputs must have 'pc_2' key" in str(exc_info.value)


def test_validate_3dcd_structure_empty_pc():
    """Test 3D change detection validation with empty point cloud dict."""
    invalid_datapoint = {
        'inputs': {
            'pc_1': {},  # Empty dict
            'pc_2': {'pos': torch.randn(100, 3)}
        },
        'labels': {'change_map': torch.randint(0, 2, (100,))},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_3dcd_structure(invalid_datapoint)
    
    assert "inputs['pc_1'] dict must not be empty" in str(exc_info.value)


def test_validate_3dcd_structure_invalid_pos_shape():
    """Test 3D change detection validation with invalid pos shape."""
    invalid_datapoint = {
        'inputs': {
            'pc_1': {'pos': torch.randn(100, 2)},  # Should be Nx3
            'pc_2': {'pos': torch.randn(100, 3)}
        },
        'labels': {'change_map': torch.randint(0, 2, (100,))},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_3dcd_structure(invalid_datapoint)
    
    assert "inputs['pc_1']['pos'] must have shape [N,3]" in str(exc_info.value)


def test_validate_3dcd_structure_dimension_mismatch():
    """Test 3D change detection validation with point cloud dimension mismatch."""
    invalid_datapoint = {
        'inputs': {
            'pc_1': {'pos': torch.randn(100, 3)},
            'pc_2': {'pos': torch.randn(100, 2)}  # Different dimension
        },
        'labels': {'change_map': torch.randint(0, 2, (100,))},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_3dcd_structure(invalid_datapoint)
    
    # The validation actually catches the invalid shape first, so check for the pos shape error
    assert "inputs['pc_2']['pos'] must have shape [N,3]" in str(exc_info.value)


def test_validate_3dcd_structure_invalid_change_map_dimensions():
    """Test 3D change detection validation with invalid change map dimensions."""
    invalid_datapoint = {
        'inputs': {
            'pc_1': {'pos': torch.randn(100, 3)},
            'pc_2': {'pos': torch.randn(100, 3)}
        },
        'labels': {'change_map': torch.randint(0, 2, (100, 1))},  # Should be 1D
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_3dcd_structure(invalid_datapoint)
    
    assert "labels['change_map'] must be 1D tensor [N]" in str(exc_info.value)


# Point cloud registration validation tests


def test_validate_pcr_structure_valid_single(valid_pcr_datapoint):
    """Test PCR validation with valid single format datapoint."""
    # Should not raise any exception
    validate_pcr_structure(valid_pcr_datapoint)


def test_validate_pcr_structure_valid_batched(valid_pcr_batched_datapoint):
    """Test PCR validation with valid batched format datapoint."""
    # Should not raise any exception
    validate_pcr_structure(valid_pcr_batched_datapoint)


def test_validate_pcr_structure_missing_src_pc():
    """Test PCR validation with missing source point cloud."""
    invalid_datapoint = {
        'inputs': {'tgt_pc': {'pos': torch.randn(100, 3)}},  # Missing src_pc
        'labels': {'transform': torch.eye(4)},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_pcr_structure(invalid_datapoint)
    
    assert "inputs must have 'src_pc' key" in str(exc_info.value)


def test_validate_pcr_structure_invalid_correspondences_shape():
    """Test PCR validation with invalid correspondences shape."""
    invalid_datapoint = {
        'inputs': {
            'src_pc': {'pos': torch.randn(100, 3)},
            'tgt_pc': {'pos': torch.randn(100, 3)},
            'correspondences': torch.randint(0, 100, (50,))  # Should be 2D
        },
        'labels': {'transform': torch.eye(4)},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_pcr_structure(invalid_datapoint)
    
    assert "inputs['correspondences'] must be 2D tensor" in str(exc_info.value)


def test_validate_pcr_structure_batched_missing_lengths():
    """Test PCR validation with batched format missing lengths."""
    invalid_datapoint = {
        'inputs': {
            'points': torch.randn(1000, 3),
            # Missing lengths/stack_lengths
        },
        'labels': {'transform': torch.eye(4)},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_pcr_structure(invalid_datapoint)
    
    assert "inputs must have 'src_pc' key" in str(exc_info.value)  # Falls back to single format validation


def test_validate_pcr_structure_invalid_transform_shape():
    """Test PCR validation with invalid transform shape."""
    invalid_datapoint = {
        'inputs': {
            'src_pc': {'pos': torch.randn(100, 3)},
            'tgt_pc': {'pos': torch.randn(100, 3)}
        },
        'labels': {'transform': torch.eye(3)},  # Should be 4x4
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_pcr_structure(invalid_datapoint)
    
    assert "labels['transform'] must have shape [4,4]" in str(exc_info.value)


# validate_structure_for_type tests


@pytest.mark.parametrize("dataset_type,datapoint_fixture", [
    ('semseg', 'valid_semseg_datapoint'),
    ('2dcd', 'valid_2dcd_datapoint'),
    ('3dcd', 'valid_3dcd_datapoint'),
    ('pcr', 'valid_pcr_datapoint'),
])
def test_validate_structure_for_type_valid(dataset_type, datapoint_fixture, request):
    """Test validate_structure_for_type with valid datapoints for each type."""
    datapoint = request.getfixturevalue(datapoint_fixture)
    
    # Should not raise any exception
    validate_structure_for_type(dataset_type, datapoint)


def test_validate_structure_for_type_invalid_type():
    """Test validate_structure_for_type with invalid dataset type."""
    datapoint = {
        'inputs': {'data': torch.randn(10)},
        'labels': {'target': torch.tensor(1)},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError) as exc_info:
        validate_structure_for_type('invalid_type', datapoint)
    
    assert "Unsupported dataset type for validation: invalid_type" in str(exc_info.value)
    assert "Supported types:" in str(exc_info.value)


@pytest.mark.parametrize("dataset_type,invalid_datapoint", [
    ('semseg', {'inputs': {'wrong_key': torch.randn(3, 32, 32)}, 'labels': {'label': torch.randint(0, 10, (32, 32))}, 'meta_info': {}}),
    ('2dcd', {'inputs': {'img_1': torch.randn(3, 32, 32)}, 'labels': {'change_map': torch.randint(0, 2, (32, 32))}, 'meta_info': {}}),
    ('3dcd', {'inputs': {'pc_1': {'pos': torch.randn(100, 3)}}, 'labels': {'change_map': torch.randint(0, 2, (100,))}, 'meta_info': {}}),
    ('pcr', {'inputs': {'src_pc': {'pos': torch.randn(100, 3)}}, 'labels': {'transform': torch.eye(4)}, 'meta_info': {}}),
])
def test_validate_structure_for_type_validation_fails(dataset_type, invalid_datapoint):
    """Test validate_structure_for_type properly delegates validation failures."""
    with pytest.raises(AssertionError):
        validate_structure_for_type(dataset_type, invalid_datapoint)