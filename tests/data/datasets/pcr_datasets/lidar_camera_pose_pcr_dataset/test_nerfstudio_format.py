"""Test nerfstudio format validation for LiDAR camera pose PCR dataset."""
import os
import json
import pytest
from data.datasets.pcr_datasets.lidar_camera_pose_pcr_dataset import LiDARCameraPosePCRDataset


def test_missing_frames_key(test_data, basic_dataset_kwargs):
    """Test validation when transforms.json is missing 'frames' key."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Create invalid transforms.json file - missing 'frames' key
    invalid_data = {"camera_model": "PINHOLE"}
    invalid_path = os.path.join(temp_dir, "invalid_missing_frames.json")
    with open(invalid_path, 'w') as f:
        json.dump(invalid_data, f)
    
    # Test initialization with invalid JSON - should fail
    with pytest.raises(AssertionError, match="transforms.json must have 'frames' key"):
        LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths[:1],
            transforms_json_filepaths=[invalid_path],
            **basic_dataset_kwargs
        )


def test_empty_frames(test_data, basic_dataset_kwargs):
    """Test validation when transforms.json has empty 'frames' array."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Create invalid transforms.json file - empty frames
    invalid_data = {"frames": []}
    invalid_path = os.path.join(temp_dir, "invalid_empty_frames.json")
    with open(invalid_path, 'w') as f:
        json.dump(invalid_data, f)
    
    # Test initialization with invalid JSON - should fail
    with pytest.raises(AssertionError, match="'frames' must not be empty"):
        LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths[:1],
            transforms_json_filepaths=[invalid_path],
            **basic_dataset_kwargs
        )


def test_invalid_json_structure(test_data, basic_dataset_kwargs):
    """Test validation of invalid JSON structure (not a dictionary)."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Create invalid transforms.json file - array instead of dict
    invalid_data = ["not", "a", "dictionary"]
    invalid_path = os.path.join(temp_dir, "invalid_structure.json")
    with open(invalid_path, 'w') as f:
        json.dump(invalid_data, f)
    
    # Test initialization with invalid JSON - should fail
    with pytest.raises(AssertionError, match="transforms.json must be a dictionary"):
        LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths[:1],
            transforms_json_filepaths=[invalid_path],
            **basic_dataset_kwargs
        )


def test_invalid_frames_type(test_data, basic_dataset_kwargs):
    """Test validation when 'frames' is not a list."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Create invalid transforms.json file - frames is not a list
    invalid_data = {"frames": {"not": "a list"}}
    invalid_path = os.path.join(temp_dir, "invalid_frames_type.json")
    with open(invalid_path, 'w') as f:
        json.dump(invalid_data, f)
    
    # Test initialization with invalid JSON - should fail
    with pytest.raises(AssertionError, match="'frames' must be a list"):
        LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths[:1],
            transforms_json_filepaths=[invalid_path],
            **basic_dataset_kwargs
        )


def test_missing_transform_matrix(test_data, basic_dataset_kwargs):
    """Test validation when frame is missing 'transform_matrix' key."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Create invalid transforms.json file - frame missing transform_matrix
    invalid_data = {
        "frames": [
            {
                "file_path": "frame_0000",
                # Missing transform_matrix
            }
        ]
    }
    invalid_path = os.path.join(temp_dir, "invalid_missing_transform.json")
    with open(invalid_path, 'w') as f:
        json.dump(invalid_data, f)
    
    # Test initialization with invalid JSON - should fail
    with pytest.raises(AssertionError, match="Frame 0 must have 'transform_matrix' key"):
        LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths[:1],
            transforms_json_filepaths=[invalid_path],
            **basic_dataset_kwargs
        )


def test_invalid_transform_matrix_shape(test_data, basic_dataset_kwargs):
    """Test validation when transform_matrix has wrong shape."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Create invalid transforms.json file - wrong transform_matrix shape
    invalid_data = {
        "frames": [
            {
                "file_path": "frame_0000",
                "transform_matrix": [
                    [1, 0, 0],  # Wrong shape - should be 4x4
                    [0, 1, 0],
                    [0, 0, 1]
                ]
            }
        ]
    }
    invalid_path = os.path.join(temp_dir, "invalid_transform_shape.json")
    with open(invalid_path, 'w') as f:
        json.dump(invalid_data, f)
    
    # Test initialization with invalid JSON - should fail
    with pytest.raises(AssertionError, match="Frame 0 transform_matrix must have 4 rows"):
        LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths[:1],
            transforms_json_filepaths=[invalid_path],
            **basic_dataset_kwargs
        )


def test_invalid_last_row(test_data, basic_dataset_kwargs):
    """Test validation when transform_matrix has invalid last row."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Create invalid transforms.json file - wrong last row
    invalid_data = {
        "frames": [
            {
                "file_path": "frame_0000",
                "transform_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1, 1, 1, 1]  # Wrong last row - should be [0, 0, 0, 1]
                ]
            }
        ]
    }
    invalid_path = os.path.join(temp_dir, "invalid_last_row.json")
    with open(invalid_path, 'w') as f:
        json.dump(invalid_data, f)
    
    # Test initialization with invalid JSON - should fail
    with pytest.raises(AssertionError, match="Frame 0 invalid last row"):
        LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths[:1],
            transforms_json_filepaths=[invalid_path],
            **basic_dataset_kwargs
        )