"""Test input validation for LiDAR camera pose PCR dataset."""
import pytest
from data.datasets.pcr_datasets.lidar_camera_pose_pcr_dataset import LiDARCameraPosePCRDataset


def test_mismatched_list_lengths(test_data, basic_dataset_kwargs):
    """Test validation of mismatched PC and JSON file list lengths."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Test mismatched list lengths
    with pytest.raises(AssertionError, match="must match number of transforms.json files"):
        LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths[:1],  # Only 1 PC file
            transforms_json_filepaths=transforms_json_filepaths,  # 2 JSON files
            **basic_dataset_kwargs
        )


def test_empty_file_lists(basic_dataset_kwargs):
    """Test validation of empty file lists."""
    # Test empty lists
    with pytest.raises(AssertionError, match="Must provide at least one point cloud file"):
        LiDARCameraPosePCRDataset(
            pc_filepaths=[],
            transforms_json_filepaths=[],
            **basic_dataset_kwargs
        )


def test_invalid_pc_filepaths_type(test_data, basic_dataset_kwargs):
    """Test validation of invalid pc_filepaths parameter type."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Test invalid file types
    with pytest.raises(AssertionError, match="pc_filepaths must be list"):
        LiDARCameraPosePCRDataset(
            pc_filepaths="not_a_list",
            transforms_json_filepaths=transforms_json_filepaths,
            **basic_dataset_kwargs
        )


def test_invalid_transforms_json_filepaths_type(test_data, basic_dataset_kwargs):
    """Test validation of invalid transforms_json_filepaths parameter type."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Test invalid transforms_json_filepaths type
    with pytest.raises(AssertionError, match="transforms_json_filepaths must be list"):
        LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths,
            transforms_json_filepaths="not_a_list",
            **basic_dataset_kwargs
        )