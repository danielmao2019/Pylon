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
    
    # Test that scene camera poses were loaded correctly
    assert len(dataset.scene_camera_poses) == 2  # Two scenes
    total_poses = sum(len(poses) for poses in dataset.scene_camera_poses.values())
    assert total_poses == 7  # 3 poses from first file + 4 from second file
    
    # Test that camera poses have correct structure
    for scene_filepath, poses in dataset.scene_camera_poses.items():
        assert isinstance(poses, list)
        assert len(poses) > 0
        for pose in poses:
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
        lidar_vertical_fov=90.0,  # Total angle: (-45.0, 45.0) → 90.0
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
    assert dataset.lidar_vertical_fov == 90.0


def test_camera_pose_loading_by_scene(test_data, basic_dataset_kwargs):
    """Test that camera poses are organized by scene correctly."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    dataset = LiDARCameraPosePCRDataset(
        pc_filepaths=pc_filepaths,
        transforms_json_filepaths=transforms_json_filepaths,
        **basic_dataset_kwargs
    )
    
    # Verify scene organization
    assert len(dataset.scene_camera_poses) == 2  # Two scenes
    expected_total = 7  # 3 from first file + 4 from second file
    total_poses = sum(len(poses) for poses in dataset.scene_camera_poses.values())
    assert total_poses == expected_total
    
    # Verify each scene has the correct number of poses
    pose_counts = [len(poses) for poses in dataset.scene_camera_poses.values()]
    pose_counts.sort()  # Sort to make assertion order-independent
    assert pose_counts == [3, 4]  # Expected counts from conftest.py
    
    # Verify each pose is a valid 4x4 matrix
    pose_idx = 0
    for scene_filepath, poses in dataset.scene_camera_poses.items():
        assert isinstance(scene_filepath, str)
        assert scene_filepath in pc_filepaths
        
        for pose in poses:
            assert pose.shape == (4, 4), f"Pose {pose_idx} has invalid shape: {pose.shape}"
            assert pose.dtype == np.float32, f"Pose {pose_idx} has invalid dtype: {pose.dtype}"
            
            # Check that it's a valid transformation matrix
            assert np.allclose(pose[3, :], [0, 0, 0, 1], atol=1e-5), f"Pose {pose_idx} has invalid bottom row: {pose[3, :]}"
            
            # Check that rotation part is orthogonal
            rotation = pose[:3, :3]
            should_be_identity = np.dot(rotation, rotation.T)
            identity = np.eye(3)
            assert np.allclose(should_be_identity, identity, atol=1e-4), f"Pose {pose_idx} rotation is not orthogonal"
            
            pose_idx += 1


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
