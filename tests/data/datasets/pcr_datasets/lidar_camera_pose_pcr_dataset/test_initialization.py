"""Test initialization functionality for LiDAR camera pose PCR dataset."""
import numpy as np
from data.datasets.pcr_datasets.lidar_camera_pose_pcr_dataset import LiDARCameraPosePCRDataset


def test_dataset_initialization_basic(test_data, basic_dataset_kwargs):
    """Test basic dataset initialization with camera pose loading."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    dataset = LiDARCameraPosePCRDataset(
        pc_filepaths=pc_filepaths,
        transforms_json_filepaths=transforms_json_filepaths,
        **basic_dataset_kwargs
    )
    
    # Basic checks
    assert len(dataset) == basic_dataset_kwargs['dataset_size']
    assert len(dataset.file_pair_annotations) == 2  # Two test files
    assert len(dataset.all_camera_poses) == 7  # 3 poses from first file + 4 from second file
    
    # Test that camera poses were loaded correctly
    for pose in dataset.all_camera_poses:
        assert pose.shape == (4, 4)
        # Check last row is [0, 0, 0, 1]
        assert np.allclose(pose[3, :], [0, 0, 0, 1])


def test_dataset_initialization_custom_parameters(test_data):
    """Test dataset initialization with custom parameters."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    dataset = LiDARCameraPosePCRDataset(
        pc_filepaths=pc_filepaths,
        transforms_json_filepaths=transforms_json_filepaths,
        dataset_size=10,
        rotation_mag=30.0,
        translation_mag=2.0,
        matching_radius=0.1,
        overlap_range=(0.3, 0.9),
        min_points=100,
        max_trials=100,
        # LiDAR parameters
        lidar_max_range=50.0,
        lidar_horizontal_fov=360.0,
        lidar_vertical_fov=(-45.0, 45.0),
        split='train',
    )
    
    # Verify parameters were set correctly
    assert len(dataset) == 10
    assert dataset.rotation_mag == 30.0
    assert dataset.translation_mag == 2.0
    assert dataset.matching_radius == 0.1
    assert dataset.overlap_range == (0.3, 0.9)
    assert dataset.min_points == 100
    assert dataset.lidar_max_range == 50.0
    assert dataset.lidar_horizontal_fov == 360.0
    assert dataset.lidar_vertical_fov == (-45.0, 45.0)


def test_camera_pose_loading_union(test_data, basic_dataset_kwargs):
    """Test that camera poses are loaded into a union correctly."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    dataset = LiDARCameraPosePCRDataset(
        pc_filepaths=pc_filepaths,
        transforms_json_filepaths=transforms_json_filepaths,
        **basic_dataset_kwargs
    )
    
    # Verify camera pose union
    expected_total = 7  # 3 from first file + 4 from second file
    assert len(dataset.all_camera_poses) == expected_total
    
    # Verify each pose is a valid 4x4 matrix
    for i, pose in enumerate(dataset.all_camera_poses):
        assert pose.shape == (4, 4), f"Pose {i} has invalid shape: {pose.shape}"
        assert pose.dtype == np.float32, f"Pose {i} has invalid dtype: {pose.dtype}"
        
        # Check that it's a valid transformation matrix
        assert np.allclose(pose[3, :], [0, 0, 0, 1], atol=1e-5), f"Pose {i} has invalid bottom row: {pose[3, :]}"
        
        # Check that rotation part is orthogonal
        rotation = pose[:3, :3]
        should_be_identity = np.dot(rotation, rotation.T)
        identity = np.eye(3)
        assert np.allclose(should_be_identity, identity, atol=1e-4), f"Pose {i} rotation is not orthogonal"


def test_file_pair_annotations(test_data, basic_dataset_kwargs):
    """Test that file pair annotations are created correctly."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    dataset = LiDARCameraPosePCRDataset(
        pc_filepaths=pc_filepaths,
        transforms_json_filepaths=transforms_json_filepaths,
        **basic_dataset_kwargs
    )
    
    # Check file pair annotations
    assert len(dataset.file_pair_annotations) == len(pc_filepaths)
    
    for i, annotation in enumerate(dataset.file_pair_annotations):
        assert isinstance(annotation, dict)
        assert 'src_filepath' in annotation
        assert 'tgt_filepath' in annotation
        
        # For single-temporal dataset, src and tgt should be the same
        assert annotation['src_filepath'] == pc_filepaths[i]
        assert annotation['tgt_filepath'] == pc_filepaths[i]