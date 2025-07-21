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
        camera_idx_1 = dp1['meta_info']['transform_config']['camera_pose_idx']
        camera_idx_2 = dp2['meta_info']['transform_config']['camera_pose_idx']
        assert camera_idx_1 == camera_idx_2, f"Camera pose indices differ at index {i}: {camera_idx_1} vs {camera_idx_2}"
        
        # Sensor positions should be the same (from same camera pose)
        pos_1 = dp1['meta_info']['transform_config']['sensor_position']
        pos_2 = dp2['meta_info']['transform_config']['sensor_position']
        assert np.allclose(pos_1, pos_2, atol=1e-6), f"Sensor positions differ at index {i}: {pos_1} vs {pos_2}"
        
        # Sensor orientations should be the same
        euler_1 = dp1['meta_info']['transform_config']['sensor_euler_angles']
        euler_2 = dp2['meta_info']['transform_config']['sensor_euler_angles']
        assert np.allclose(euler_1, euler_2, atol=1e-6), f"Sensor orientations differ at index {i}: {euler_1} vs {euler_2}"
        
        # Transform parameters should be the same
        rot_1 = dp1['meta_info']['transform_config']['rotation_angles']
        rot_2 = dp2['meta_info']['transform_config']['rotation_angles']
        assert np.allclose(rot_1, rot_2, atol=1e-6), f"Rotation angles differ at index {i}: {rot_1} vs {rot_2}"
        
        trans_1 = dp1['meta_info']['transform_config']['translation']
        trans_2 = dp2['meta_info']['transform_config']['translation']
        assert np.allclose(trans_1, trans_2, atol=1e-6), f"Translations differ at index {i}: {trans_1} vs {trans_2}"


def test_different_seeds_produce_different_results(test_data):
    """Test that different seeds can produce different camera pose sampling."""
    # Note: This test validates that the seeding mechanism works correctly.
    # Due to the small number of camera poses (7) and potential overlap constraints,
    # identical patterns are possible but the mechanism should still be functional.
    
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    # Test multiple seed pairs to increase confidence
    seed_pairs = [(42, 999), (1, 12345), (0, 777)]
    
    for seed1, seed2 in seed_pairs:
        # Common parameters for both datasets
        base_kwargs = {
            'dataset_size': 10,
            'rotation_mag': 5.0,
            'translation_mag': 0.5,
            'matching_radius': 0.2,
            'overlap_range': (0.1, 1.0),
            'min_points': 10,
            'max_trials': 100,
            'split': 'train',
        }
        
        # Create datasets with different seeds
        dataset1 = LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths,
            transforms_json_filepaths=transforms_json_filepaths,
            base_seed=seed1,
            **base_kwargs
        )
        
        dataset2 = LiDARCameraPosePCRDataset(
            pc_filepaths=pc_filepaths,
            transforms_json_filepaths=transforms_json_filepaths,
            base_seed=seed2,
            **base_kwargs
        )
        
        # Compare camera pose indices
        indices1 = [dataset1[i]['meta_info']['transform_config']['camera_pose_idx'] for i in range(len(dataset1))]
        indices2 = [dataset2[i]['meta_info']['transform_config']['camera_pose_idx'] for i in range(len(dataset2))]
        
        # Check if any differences exist for this seed pair
        differences = sum(1 for i1, i2 in zip(indices1, indices2) if i1 != i2)
        
        if differences > 0:
            # Found differences with this seed pair - test passes
            return
    
    # If we get here, all seed pairs produced identical results.
    # This is extremely unlikely but possible with a small pose set.
    # The important thing is that the seeding mechanism is working,
    # which we can verify by checking that rotation/translation parameters differ.
    
    # Test with just the first seed pair
    dataset1 = LiDARCameraPosePCRDataset(
        pc_filepaths=pc_filepaths,
        transforms_json_filepaths=transforms_json_filepaths,
        base_seed=42,
        dataset_size=5,
        rotation_mag=5.0,
        translation_mag=0.5,
        matching_radius=0.2,
        overlap_range=(0.1, 1.0),
        min_points=10,
        max_trials=100,
        split='train',
    )
    
    dataset2 = LiDARCameraPosePCRDataset(
        pc_filepaths=pc_filepaths,
        transforms_json_filepaths=transforms_json_filepaths,
        base_seed=999,
        dataset_size=5,
        rotation_mag=5.0,
        translation_mag=0.5,
        matching_radius=0.2,
        overlap_range=(0.1, 1.0),
        min_points=10,
        max_trials=100,
        split='train',
    )
    
    # Compare rotation angles (these should definitely be different)
    rotation_diffs = 0
    for i in range(min(len(dataset1), len(dataset2))):
        rot1 = dataset1[i]['meta_info']['transform_config']['rotation_angles']
        rot2 = dataset2[i]['meta_info']['transform_config']['rotation_angles']
        if not np.allclose(rot1, rot2, atol=1e-6):
            rotation_diffs += 1
    
    assert rotation_diffs > 0, "Different seeds should produce different rotation parameters even if camera poses are the same"


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