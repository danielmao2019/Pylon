"""Shared fixtures and helper functions for KITTI dataset tests."""

import pytest
import os
import numpy as np
from contextlib import contextmanager
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset


@pytest.fixture
def create_dummy_kitti_structure():
    """Fixture that returns a function to create dummy KITTI directory structure for testing."""
    def _create_structure(data_root: str) -> None:
        """Create a dummy KITTI directory structure for testing."""
        # Create sequences directory
        sequences_dir = os.path.join(data_root, 'sequences')
        os.makedirs(sequences_dir, exist_ok=True)
        
        # Create poses directory
        poses_dir = os.path.join(data_root, 'poses')
        os.makedirs(poses_dir, exist_ok=True)
        
        # Create dummy sequences (all sequences needed for KITTI splits)
        test_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        
        for seq in test_sequences:
            # Create velodyne directory for point cloud files
            velodyne_dir = os.path.join(sequences_dir, seq, 'velodyne')
            os.makedirs(velodyne_dir, exist_ok=True)
            
            # Create dummy .bin files (5 files per sequence)
            for i in range(5):
                bin_file = os.path.join(velodyne_dir, f'{i:06d}.bin')
                # Create dummy point cloud data (4 points with x,y,z,reflectance)
                dummy_data = np.array([
                    [i*1.0, 0.0, 0.0, 0.5],
                    [i*1.0+1.0, 0.0, 0.0, 0.5],
                    [i*1.0, 1.0, 0.0, 0.5],
                    [i*1.0, 0.0, 1.0, 0.5]
                ], dtype=np.float32)
                dummy_data.tofile(bin_file)
            
            # Create dummy pose file
            pose_file = os.path.join(poses_dir, f'{seq}.txt')
            with open(pose_file, 'w') as f:
                # Write 5 poses (12 values each: 3x4 transformation matrix flattened)
                for i in range(5):
                    # Identity transformation with different translation for each frame
                    pose = [1.0, 0.0, 0.0, i*10.0,  # first row: [R|t]
                           0.0, 1.0, 0.0, 0.0,      # second row
                           0.0, 0.0, 1.0, 0.0]      # third row
                    f.write(' '.join(map(str, pose)) + '\n')
    
    return _create_structure


@pytest.fixture
def patched_kitti_dataset_size():
    """Fixture that patches KITTI DATASET_SIZE for testing."""
    @contextmanager
    def _patch():
        """Context manager to temporarily patch DATASET_SIZE for testing."""
        original_dataset_size = KITTIDataset.DATASET_SIZE
        # Calculate actual sizes based on 5 files per sequence
        KITTIDataset.DATASET_SIZE = {
            'train': 12,  # 6 sequences * 2 pairs per sequence = 12 pairs  
            'val': 4,     # 2 sequences * 2 pairs per sequence = 4 pairs
            'test': 6,    # 3 sequences * 2 pairs per sequence = 6 pairs
        }
        try:
            yield
        finally:
            KITTIDataset.DATASET_SIZE = original_dataset_size
    
    return _patch