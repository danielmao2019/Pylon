"""Tests for SingleTemporalPCRDataset cache version discrimination."""

import pytest
import tempfile
import os
from data.datasets.pcr_datasets.single_temporal_pcr_dataset import SingleTemporalPCRDataset


def create_dummy_point_cloud_files(data_root: str, num_files: int = 4) -> None:
    """Create dummy point cloud files for testing."""
    for i in range(num_files):
        # Create dummy PLY files
        ply_file = os.path.join(data_root, f'pointcloud_{i:04d}.ply')
        with open(ply_file, 'w') as f:
            f.write(f'ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n{i}.0 0.0 0.0\n{i+1}.0 0.0 0.0\n{i}.0 1.0 0.0\n')


def test_single_temporal_pcr_dataset_version_discrimination():
    """Test that SingleTemporalPCRDataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_point_cloud_files(temp_dir)
        
        # Same parameters should have same hash
        dataset1a = SingleTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05
        )
        dataset1b = SingleTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different dataset_size should have different hash
        dataset2 = SingleTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=200,  # Different
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different rotation_mag should have different hash
        dataset3 = SingleTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=30.0,  # Different
            translation_mag=0.5,
            matching_radius=0.05
        )
        assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()
        
        # Different translation_mag should have different hash
        dataset4 = SingleTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.3,  # Different
            matching_radius=0.05
        )
        assert dataset1a.get_cache_version_hash() != dataset4.get_cache_version_hash()
        
        # Different matching_radius should have different hash
        dataset5 = SingleTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.1  # Different
        )
        assert dataset1a.get_cache_version_hash() != dataset5.get_cache_version_hash()


def test_inherited_parameters_affect_version_hash():
    """Test that parameters inherited from SyntheticTransformPCRDataset affect version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_point_cloud_files(temp_dir)
        
        base_args = {
            'data_root': temp_dir,
            'dataset_size': 100,
        }
        
        # Test inherited parameters from SyntheticTransformPCRDataset
        parameter_variants = [
            ('rotation_mag', 30.0),  # Different from default 45.0
            ('translation_mag', 0.3),  # Different from default 0.5
            ('matching_radius', 0.1),  # Different from default 0.05
            ('overlap_range', (0.2, 0.9)),  # Different from default (0.3, 1.0)
            ('min_points', 256),  # Different from default 512
            ('max_trials', 500),  # Different from default 1000
            ('lidar_max_range', 8.0),  # Different from default 6.0
            ('lidar_horizontal_fov', 90.0),  # Different from default 120.0
            ('lidar_vertical_fov', 45.0),  # Different from default 60.0
            ('lidar_apply_range_filter', True),  # Different from default False
            ('lidar_apply_fov_filter', False),  # Different from default True
            ('lidar_apply_occlusion_filter', True),  # Different from default False
        ]
        
        dataset1 = SingleTemporalPCRDataset(**base_args)
        
        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = SingleTemporalPCRDataset(**modified_args)
            
            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Inherited parameter {param_name} should affect cache version hash"


def test_different_point_cloud_files():
    """Test that different sets of point cloud files produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create first set of files
        data_dir1 = os.path.join(temp_dir, 'data1')
        os.makedirs(data_dir1)
        create_dummy_point_cloud_files(data_dir1, num_files=3)
        
        # Create second set of files (different count)
        data_dir2 = os.path.join(temp_dir, 'data2')
        os.makedirs(data_dir2)
        create_dummy_point_cloud_files(data_dir2, num_files=5)
        
        dataset1 = SingleTemporalPCRDataset(
            data_root=data_dir1,
            dataset_size=100
        )
        dataset2 = SingleTemporalPCRDataset(
            data_root=data_dir2,
            dataset_size=100
        )
        
        # Different file sets should produce different hashes
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_different_file_types():
    """Test that datasets with different point cloud file types are discriminated."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create first set with PLY files
        data_dir1 = os.path.join(temp_dir, 'data1')
        os.makedirs(data_dir1)
        for i in range(3):
            ply_file = os.path.join(data_dir1, f'pointcloud_{i:04d}.ply')
            with open(ply_file, 'w') as f:
                f.write('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n')
        
        # Create second set with OFF files
        data_dir2 = os.path.join(temp_dir, 'data2')
        os.makedirs(data_dir2)
        for i in range(3):
            off_file = os.path.join(data_dir2, f'pointcloud_{i:04d}.off')
            with open(off_file, 'w') as f:
                f.write('OFF\n3 1 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n3 0 1 2\n')
        
        dataset1 = SingleTemporalPCRDataset(
            data_root=data_dir1,
            dataset_size=100
        )
        dataset2 = SingleTemporalPCRDataset(
            data_root=data_dir2,
            dataset_size=100
        )
        
        # Different file types should produce different hashes
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_lidar_parameters_variants():
    """Test that different LiDAR simulation parameters are properly discriminated."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_point_cloud_files(temp_dir)
        
        lidar_parameter_sets = [
            # Standard LiDAR setup
            {
                'lidar_max_range': 6.0,
                'lidar_horizontal_fov': 120.0,
                'lidar_vertical_fov': 60.0,
                'lidar_apply_range_filter': False,
                'lidar_apply_fov_filter': True,
                'lidar_apply_occlusion_filter': False
            },
            # Extended range LiDAR
            {
                'lidar_max_range': 10.0,
                'lidar_horizontal_fov': 120.0,
                'lidar_vertical_fov': 60.0,
                'lidar_apply_range_filter': True,
                'lidar_apply_fov_filter': True,
                'lidar_apply_occlusion_filter': False
            },
            # Wide FOV LiDAR
            {
                'lidar_max_range': 6.0,
                'lidar_horizontal_fov': 360.0,
                'lidar_vertical_fov': 90.0,
                'lidar_apply_range_filter': False,
                'lidar_apply_fov_filter': True,
                'lidar_apply_occlusion_filter': True
            },
            # All filters disabled
            {
                'lidar_max_range': 6.0,
                'lidar_horizontal_fov': 120.0,
                'lidar_vertical_fov': 60.0,
                'lidar_apply_range_filter': False,
                'lidar_apply_fov_filter': False,
                'lidar_apply_occlusion_filter': False
            }
        ]
        
        datasets = []
        for lidar_params in lidar_parameter_sets:
            dataset = SingleTemporalPCRDataset(
                data_root=temp_dir,
                dataset_size=100,
                **lidar_params
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All LiDAR parameter variants should produce different hashes, got: {hashes}"


def test_overlap_range_variants():
    """Test that different overlap ranges are properly discriminated."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_point_cloud_files(temp_dir)
        
        overlap_range_variants = [
            (0.3, 1.0),  # Default
            (0.2, 0.8),  # Smaller range
            (0.4, 1.0),  # Higher minimum
            (0.3, 0.9),  # Lower maximum
            (0.5, 0.7),  # Narrow range
        ]
        
        datasets = []
        for overlap_range in overlap_range_variants:
            dataset = SingleTemporalPCRDataset(
                data_root=temp_dir,
                dataset_size=100,
                overlap_range=overlap_range
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All overlap range variants should produce different hashes, got: {hashes}"


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_point_cloud_files(temp_dir)
        
        datasets = []
        
        # Generate many different dataset configurations
        for dataset_size in [50, 100, 200]:
            for rotation_mag in [30.0, 45.0, 60.0]:
                for translation_mag in [0.3, 0.5, 0.7]:
                    for matching_radius in [0.05, 0.1]:
                        for lidar_max_range in [6.0, 8.0]:
                            datasets.append(SingleTemporalPCRDataset(
                                data_root=temp_dir,
                                dataset_size=dataset_size,
                                rotation_mag=rotation_mag,
                                translation_mag=translation_mag,
                                matching_radius=matching_radius,
                                lidar_max_range=lidar_max_range
                            ))
        
        # Collect all hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        
        # Ensure all hashes are unique (no collisions)
        assert len(hashes) == len(set(hashes)), \
            f"Hash collision detected! Duplicate hashes found in: {hashes}"
        
        # Ensure all hashes are properly formatted
        for hash_val in hashes:
            assert isinstance(hash_val, str), f"Hash must be string, got {type(hash_val)}"
            assert len(hash_val) == 16, f"Hash must be 16 characters, got {len(hash_val)}"


