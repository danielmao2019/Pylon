"""Fixtures and utilities for LiDAR camera pose PCR dataset tests."""
import os
import json
import tempfile
import pytest
import numpy as np
from typing import Tuple, List


@pytest.fixture
def test_data():
    """Create temporary test data files for LiDAR camera pose PCR dataset tests."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create test point cloud files
    pc_filepaths = []
    transforms_json_filepaths = []
    
    for i in range(2):  # Create 2 test files
        # Create a simple test point cloud (saved as .txt for simplicity)
        num_points = 500 + i * 100  # Different sizes for variety
        points = np.random.randn(num_points, 3).astype(np.float32) * 5.0
        pc_filepath = os.path.join(temp_dir, f"test_cloud_{i}.txt")
        np.savetxt(pc_filepath, points, fmt='%.6f')
        pc_filepaths.append(pc_filepath)
        
        # Create transforms.json with sample camera poses (nerfstudio format)
        camera_poses = []
        num_poses = 3 + i  # Different numbers of poses
        
        for j in range(num_poses):
            # Create a camera pose with some variation
            angle = 2 * np.pi * j / num_poses
            
            # Camera position on a circle
            radius = 10.0
            x = radius * np.cos(angle) + np.random.randn() * 0.1
            y = radius * np.sin(angle) + np.random.randn() * 0.1
            z = 5.0 + np.random.randn() * 0.5
            
            # Look towards origin with some variation
            look_at = np.array([0, 0, 0]) + np.random.randn(3) * 0.1
            
            # Compute camera rotation matrix (look-at)
            forward = look_at - np.array([x, y, z])
            forward = forward / np.linalg.norm(forward)
            
            # Assume up is roughly [0, 0, 1]
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # Build rotation matrix
            rotation = np.eye(3)
            rotation[:, 0] = right
            rotation[:, 1] = up
            rotation[:, 2] = -forward  # OpenGL convention
            
            # Build 4x4 transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = [x, y, z]
            
            camera_poses.append({
                "file_path": f"frame_{j:04d}",
                "transform_matrix": transform.tolist()
            })
        
        # Save transforms.json (nerfstudio format)
        transforms_data = {
            "camera_model": "PINHOLE",
            "frames": camera_poses
        }
        
        json_filepath = os.path.join(temp_dir, f"transforms_{i}.json")
        with open(json_filepath, 'w') as f:
            json.dump(transforms_data, f, indent=2)
        transforms_json_filepaths.append(json_filepath)
    
    yield temp_dir, pc_filepaths, transforms_json_filepaths
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def basic_dataset_kwargs():
    """Basic kwargs for creating LiDAR camera pose PCR dataset."""
    return {
        'dataset_size': 5,
        'rotation_mag': 15.0,
        'translation_mag': 1.0,
        'matching_radius': 0.2,
        'overlap_range': (0.1, 1.0),  # Permissive for testing
        'min_points': 10,  # Lower minimum for testing
        'max_trials': 100,
        'lidar_max_range': 100.0,
        'lidar_horizontal_fov': 360.0,
        'lidar_vertical_fov': 180.0,  # Total angle: (-90.0, 90.0) → 180.0
        'split': 'train',
    }


@pytest.fixture
def dataset(test_data, basic_dataset_kwargs):
    """Create a basic LiDAR camera pose PCR dataset for testing."""
    from data.datasets.pcr_datasets.lidar_camera_pose_pcr_dataset import LiDARCameraPosePCRDataset
    
    temp_dir, pc_filepaths, transforms_json_filepaths = test_data
    
    return LiDARCameraPosePCRDataset(
        pc_filepaths=pc_filepaths,
        transforms_json_filepaths=transforms_json_filepaths,
        **basic_dataset_kwargs
    )
