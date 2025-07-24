"""Tests for BiTemporalPCRDataset cache version discrimination."""

import pytest
import tempfile
import os
import json
from data.datasets.pcr_datasets.bi_temporal_pcr_dataset import BiTemporalPCRDataset


def create_dummy_point_cloud_files(data_root: str, num_files: int = 4) -> None:
    """Create dummy point cloud files for testing."""
    for i in range(num_files):
        # Create dummy PLY files
        ply_file = os.path.join(data_root, f'pointcloud_{i:04d}.ply')
        with open(ply_file, 'w') as f:
            f.write(f'ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n{i}.0 0.0 0.0\n{i+1}.0 0.0 0.0\n{i}.0 1.0 0.0\n')


def create_dummy_gt_transforms(filepath: str) -> None:
    """Create a dummy GT transforms JSON file for testing."""
    transforms_data = {
        "pointcloud_0000.ply|pointcloud_0001.ply": [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "pointcloud_0001.ply|pointcloud_0002.ply": [
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    }
    
    with open(filepath, 'w') as f:
        json.dump(transforms_data, f)


def test_bi_temporal_pcr_dataset_version_discrimination():
    """Test that BiTemporalPCRDataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_point_cloud_files(temp_dir)
        
        # Same parameters should have same hash
        dataset1a = BiTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05
        )
        dataset1b = BiTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different dataset_size should have different hash
        dataset2 = BiTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=200,  # Different
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different rotation_mag should have different hash
        dataset3 = BiTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=30.0,  # Different
            translation_mag=0.5,
            matching_radius=0.05
        )
        assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()


def test_gt_transforms_affect_version_hash():
    """Test that gt_transforms parameter affects version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_point_cloud_files(temp_dir)
        
        # Create GT transforms file
        gt_transforms_file = os.path.join(temp_dir, 'gt_transforms.json')
        create_dummy_gt_transforms(gt_transforms_file)
        
        # Dataset without GT transforms
        dataset1 = BiTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            gt_transforms=None
        )
        
        # Dataset with GT transforms
        dataset2 = BiTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            gt_transforms=gt_transforms_file
        )
        
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_different_gt_transforms_files():
    """Test that different GT transforms files produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_point_cloud_files(temp_dir)
        
        # Create first GT transforms file
        gt_transforms_file1 = os.path.join(temp_dir, 'gt_transforms1.json')
        create_dummy_gt_transforms(gt_transforms_file1)
        
        # Create second GT transforms file with different content
        gt_transforms_file2 = os.path.join(temp_dir, 'gt_transforms2.json')
        transforms_data2 = {
            "pointcloud_0000.ply|pointcloud_0001.ply": [
                [1.0, 0.0, 0.0, 0.5],  # Different translation
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }
        with open(gt_transforms_file2, 'w') as f:
            json.dump(transforms_data2, f)
        
        # Datasets with different GT transforms should have different hashes
        dataset1 = BiTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            gt_transforms=gt_transforms_file1
        )
        dataset2 = BiTemporalPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            gt_transforms=gt_transforms_file2
        )
        
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


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
        
        dataset1 = BiTemporalPCRDataset(**base_args)
        
        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = BiTemporalPCRDataset(**modified_args)
            
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
        
        dataset1 = BiTemporalPCRDataset(
            data_root=data_dir1,
            dataset_size=100
        )
        dataset2 = BiTemporalPCRDataset(
            data_root=data_dir2,
            dataset_size=100
        )
        
        # Different file sets should produce different hashes
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_point_cloud_files(temp_dir)
        
        # Create GT transforms file
        gt_transforms_file = os.path.join(temp_dir, 'gt_transforms.json')
        create_dummy_gt_transforms(gt_transforms_file)
        
        datasets = []
        
        # Generate many different dataset configurations
        for dataset_size in [50, 100, 200]:
            for rotation_mag in [30.0, 45.0, 60.0]:
                for gt_transforms in [None, gt_transforms_file]:
                    for lidar_max_range in [6.0, 8.0]:
                        datasets.append(BiTemporalPCRDataset(
                            data_root=temp_dir,
                            dataset_size=dataset_size,
                            rotation_mag=rotation_mag,
                            gt_transforms=gt_transforms,
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


