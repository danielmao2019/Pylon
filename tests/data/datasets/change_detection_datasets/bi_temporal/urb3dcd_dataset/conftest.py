"""Shared fixtures and helper functions for urb3dcd_dataset tests."""

import pytest
import os


@pytest.fixture
def create_dummy_urb3dcd_files():
    """Fixture that returns a function for creating dummy dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create minimal URB3DCD dataset structure for testing."""
        # Create directory structure for version 1
        version_dir = os.path.join(temp_dir, 'IEEE_Dataset_V1', '1-Lidar05')
        
        for split_name in ['TrainLarge-1c', 'Val', 'Test']:
            split_dir = os.path.join(version_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Create a dummy scene
            scene_dir = os.path.join(split_dir, 'test_scene')
            os.makedirs(scene_dir, exist_ok=True)
            
            # Create epoch directories for each scene
            for epoch in ['Urb3DSimul_0001', 'Urb3DSimul_0002']:
                epoch_dir = os.path.join(scene_dir, epoch)
                os.makedirs(epoch_dir, exist_ok=True)
                
                # Create PLY files
                ply_files = ['test.ply'] if not ('train' in split_name.lower()) else ['test_patch_001.ply', 'test_patch_002.ply']
                for ply_file in ply_files:
                    ply_path = os.path.join(epoch_dir, ply_file)
                    with open(ply_path, 'w') as f:
                        f.write('ply\nformat ascii 1.0\nelement vertex 3\n')
                        f.write('property float x\nproperty float y\nproperty float z\n')
                        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
                        f.write('property int label\nend_header\n')
                        f.write('1.0 2.0 3.0 255 255 255 0\n')
                        f.write('4.0 5.0 6.0 255 255 255 1\n')
                        f.write('7.0 8.0 9.0 255 255 255 2\n')

    return _create_files