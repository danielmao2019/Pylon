"""Test deterministic behavior for LiDAR camera pose PCR dataset."""
import numpy as np
from data.datasets.pcr_datasets.lidar_camera_pose_pcr_dataset import LiDARCameraPosePCRDataset


def test_camera_pose_sampling_determinism(test_data):
    """Test that camera pose sampling is deterministic with seeds."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Common parameters for both datasets
    kwargs = {
        'dataset_size': 3,  # Smaller dataset for faster testing
        'rotation_mag': 5.0,
        'translation_mag': 0.5,
        'matching_radius': 0.2,
        'overlap_range': (0.1, 1.0),  # More permissive
        'min_points': 10,
        'max_trials': 100,
        'base_seed': 42,  # Same seed for determinism
        'split': 'train',
    }
    
    # Create two identical datasets
    dataset1 = LiDARCameraPosePCRDataset(
        pc_filepaths=pc_filepaths,
        transforms_json_filepaths=transforms_json_filepaths,
        **kwargs
    )
    
    dataset2 = LiDARCameraPosePCRDataset(
        pc_filepaths=pc_filepaths,
        transforms_json_filepaths=transforms_json_filepaths,
        **kwargs
    )
    
    # Test determinism across all datapoints
    for i in range(len(dataset1)):
        dp1 = dataset1[i]
        dp2 = dataset2[i]
        
        # Camera pose indices should be the same
        camera_idx_1 = dp1['meta_info']['transform_params']['camera_pose_idx']
        camera_idx_2 = dp2['meta_info']['transform_params']['camera_pose_idx']
        assert camera_idx_1 == camera_idx_2, f"Camera pose indices differ at index {i}: {camera_idx_1} vs {camera_idx_2}"
        
        # Sensor positions should be the same (from same camera pose)
        pos_1 = dp1['meta_info']['transform_params']['sensor_position']
        pos_2 = dp2['meta_info']['transform_params']['sensor_position']
        assert np.allclose(pos_1, pos_2, atol=1e-6), f"Sensor positions differ at index {i}: {pos_1} vs {pos_2}"
        
        # Sensor orientations should be the same
        euler_1 = dp1['meta_info']['transform_params']['sensor_euler_angles']
        euler_2 = dp2['meta_info']['transform_params']['sensor_euler_angles']
        assert np.allclose(euler_1, euler_2, atol=1e-6), f"Sensor orientations differ at index {i}: {euler_1} vs {euler_2}"
        
        # Transform parameters should be the same
        rot_1 = dp1['meta_info']['transform_params']['rotation_angles']
        rot_2 = dp2['meta_info']['transform_params']['rotation_angles']
        assert np.allclose(rot_1, rot_2, atol=1e-6), f"Rotation angles differ at index {i}: {rot_1} vs {rot_2}"
        
        trans_1 = dp1['meta_info']['transform_params']['translation']
        trans_2 = dp2['meta_info']['transform_params']['translation']
        assert np.allclose(trans_1, trans_2, atol=1e-6), f"Translations differ at index {i}: {trans_1} vs {trans_2}"




def test_dataset_length_consistency(test_data):
    """Test that dataset length is consistent across multiple instantiations."""
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    kwargs = {
        'dataset_size': 10,
        'rotation_mag': 15.0,
        'translation_mag': 1.0,
        'matching_radius': 0.2,
        'overlap_range': (0.1, 1.0),
        'min_points': 10,
        'max_trials': 100,
        'base_seed': 42,
        'split': 'train',
    }
    
    # Create multiple dataset instances
    datasets = []
    for _ in range(3):
        dataset = LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths,
            transforms_json_filepaths=transforms_json_filepaths,
            **kwargs
        )
        datasets.append(dataset)
    
    # All datasets should have the same length
    lengths = [len(dataset) for dataset in datasets]
    assert all(length == lengths[0] for length in lengths), f"Dataset lengths should be consistent: {lengths}"
    
    # All datasets should have the same number of camera poses
    camera_pose_counts = [len(dataset.all_camera_poses) for dataset in datasets]
    assert all(count == camera_pose_counts[0] for count in camera_pose_counts), f"Camera pose counts should be consistent: {camera_pose_counts}"