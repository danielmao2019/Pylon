"""Shared fixtures and helper functions for KC-3D dataset tests."""

import pytest
import os
import pickle
import numpy as np
from PIL import Image
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset


@pytest.fixture
def kc_3d_dataset_train_config(kc_3d_data_root, use_cpu_device, get_device):
    """Fixture for creating a KC3DDataset config with train split."""
    return {
        'class': KC3DDataset,
        'args': {
            'data_root': kc_3d_data_root,
            'split': 'train',
            'device': get_device(use_cpu_device)
        }
    }


@pytest.fixture
def dataset_config(request, kc_3d_data_root, use_cpu_device, get_device):
    """Fixture for creating a KC3DDataset config with parameterized split."""
    split = request.param
    return {
        'class': KC3DDataset,
        'args': {
            'data_root': kc_3d_data_root,
            'split': split,
            'device': get_device(use_cpu_device)
        }
    }


@pytest.fixture
def create_dummy_kc3d_files():
    """Fixture that provides function to create dummy KC-3D dataset files in a directory."""
    def _create_dummy_kc3d_files(data_root: str):
        """Create minimal KC-3D dataset structure for testing."""
        # Create directory structure
        os.makedirs(data_root, exist_ok=True)
        
        # Create dummy image files (RGB images with alpha channel - KC3D expects 4 channels)
        dummy_image = np.zeros((100, 100, 4), dtype=np.uint8)
        dummy_image[:, :, :3] = 128  # Gray RGB
        dummy_image[:, :, 3] = 255   # Full alpha
        
        # Create dummy mask files (binary masks)
        dummy_mask = np.zeros((100, 100), dtype=np.uint8)
        dummy_mask[20:80, 20:80] = 255  # White square in center
        
        # Create dummy depth files (single channel depth)
        dummy_depth = np.ones((100, 100), dtype=np.float32) * 1.5  # 1.5m depth
        
        # Create sample files
        for i in range(3):  # 3 samples for each split
            for split in ['train', 'val', 'test']:
                # Create sample directory if needed
                sample_dir = os.path.join(data_root, split)
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save image files (4-channel RGBA)
                img1_path = os.path.join(sample_dir, f"sample_{i}_img1.png")
                img2_path = os.path.join(sample_dir, f"sample_{i}_img2.png")
                Image.fromarray(dummy_image, mode='RGBA').save(img1_path)
                Image.fromarray(dummy_image, mode='RGBA').save(img2_path)
                
                # Save mask files (grayscale)
                mask1_path = os.path.join(sample_dir, f"sample_{i}_mask1.png")
                mask2_path = os.path.join(sample_dir, f"sample_{i}_mask2.png")
                Image.fromarray(dummy_mask, mode='L').save(mask1_path)
                Image.fromarray(dummy_mask, mode='L').save(mask2_path)
                
                # Save depth files (single channel, save as numpy arrays)
                depth1_path = os.path.join(sample_dir, f"sample_{i}_depth1.npy")
                depth2_path = os.path.join(sample_dir, f"sample_{i}_depth2.npy")
                np.save(depth1_path, dummy_depth)
                np.save(depth2_path, dummy_depth)
        
        # Create data split annotations
        annotations = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for split in ['train', 'val', 'test']:
            for i in range(3):
                annotations[split].append({
                    'image1': f"{split}/sample_{i}_img1.png",
                    'image2': f"{split}/sample_{i}_img2.png", 
                    'mask1': f"{split}/sample_{i}_mask1.png",
                    'mask2': f"{split}/sample_{i}_mask2.png",
                    'depth1': f"{split}/sample_{i}_depth1.npy",
                    'depth2': f"{split}/sample_{i}_depth2.npy"
                })
        
        # Save data split file
        split_file_path = os.path.join(data_root, "data_split.pkl")
        with open(split_file_path, "wb") as f:
            pickle.dump(annotations, f)
        
        # Create sample metadata files for ground truth registration
        dummy_metadata = {
            'intrinsics': [[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]],
            'position_before': [0.0, 0.0, 0.0],
            'position_after': [0.1, 0.0, 0.0], 
            'rotation_before': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'rotation_after': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        }
        
        for split in ['train', 'val', 'test']:
            for i in range(3):
                metadata_path = os.path.join(data_root, f"{split}_sample_{i}.npy") 
                np.save(metadata_path, dummy_metadata, allow_pickle=True)
    
    return _create_dummy_kc3d_files
