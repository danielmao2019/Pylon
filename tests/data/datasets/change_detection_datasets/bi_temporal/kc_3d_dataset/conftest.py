"""Shared fixtures and helper functions for KC3DDataset tests."""

import pytest
import tempfile
import os
import pickle
import numpy as np
from PIL import Image
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset


@pytest.fixture
def create_dummy_kc3d_files():
    """Fixture that returns a function to create minimal KC3D dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create minimal KC3D dataset structure for testing."""
        # Create dummy data split
        dummy_splits = {
            'train': [
                {
                    'image1': 'images/train_img1_before.png',
                    'image2': 'images/train_img1_after.png',
                    'depth1': 'depth/train_depth1_before.png',
                    'depth2': 'depth/train_depth1_after.png',
                    'mask1': 'masks/train_mask1_before.png',
                    'mask2': 'masks/train_mask1_after.png',
                },
                {
                    'image1': 'images/train_img2_before.png',
                    'image2': 'images/train_img2_after.png',
                    'depth1': 'depth/train_depth2_before.png',
                    'depth2': 'depth/train_depth2_after.png',
                    'mask1': 'masks/train_mask2_before.png',
                    'mask2': 'masks/train_mask2_after.png',
                },
            ],
            'val': [
                {
                    'image1': 'images/val_img1_before.png',
                    'image2': 'images/val_img1_after.png',
                    'depth1': 'depth/val_depth1_before.png',
                    'depth2': 'depth/val_depth1_after.png',
                    'mask1': 'masks/val_mask1_before.png',
                    'mask2': 'masks/val_mask1_after.png',
                },
            ],
            'test': [
                {
                    'image1': 'images/test_img1_before.png',
                    'image2': 'images/test_img1_after.png',
                    'depth1': 'depth/test_depth1_before.png',
                    'depth2': 'depth/test_depth1_after.png',
                    'mask1': 'masks/test_mask1_before.png',
                    'mask2': 'masks/test_mask1_after.png',
                },
            ],
        }
        
        # Save data split file
        with open(os.path.join(temp_dir, 'data_split.pkl'), 'wb') as f:
            pickle.dump(dummy_splits, f)
        
        # Create directory structure
        os.makedirs(os.path.join(temp_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'masks'), exist_ok=True)
        
        # Create all required files for each split
        for split_name, split_data in dummy_splits.items():
            for ann in split_data:
                # Create RGB images (4 channels, expecting RGBA)
                rgba_image = Image.new('RGBA', (64, 64), color=(128, 128, 128, 255))
                rgba_image.save(os.path.join(temp_dir, ann['image1']))
                rgba_image.save(os.path.join(temp_dir, ann['image2']))
                
                # Create depth images (single channel)
                depth_image = Image.new('L', (64, 64), color=128)
                depth_image.save(os.path.join(temp_dir, ann['depth1']))
                depth_image.save(os.path.join(temp_dir, ann['depth2']))
                
                # Create mask images (binary masks for bounding box generation)
                mask_image = Image.new('L', (64, 64), color=0)
                # Add a small white rectangle in the mask for bounding box detection
                pixels = mask_image.load()
                for x in range(20, 40):
                    for y in range(20, 40):
                        pixels[x, y] = 255
                mask_image.save(os.path.join(temp_dir, ann['mask1']))
                mask_image.save(os.path.join(temp_dir, ann['mask2']))
                
                # Create metadata files for ground truth registration
                # Extract base name for metadata file
                base_name = "_".join(ann['image1'].split(".")[0].split("_")[:3])
                metadata_path = os.path.join(temp_dir, f"{base_name}.npy")
                
                # Create dummy metadata
                metadata = {
                    'intrinsics': np.eye(3, dtype=np.float32),
                    'position_before': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    'position_after': np.array([0.1, 0.0, 0.0], dtype=np.float32),
                    'rotation_before': np.eye(3, dtype=np.float32),
                    'rotation_after': np.array([[0.9, -0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                }
                
                np.save(metadata_path, metadata)
    
    return _create_files


@pytest.fixture
def kc_3d_data_root(create_dummy_kc3d_files):
    """Fixture that creates a temporary directory with dummy KC3D dataset."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kc3d_files(temp_dir)
        yield temp_dir


@pytest.fixture
def kc_3d_dataset_train(kc_3d_data_root):
    """Fixture for creating a KC3DDataset instance with train split."""
    return KC3DDataset(data_root=kc_3d_data_root, split='train')


@pytest.fixture
def dataset(request, kc_3d_data_root):
    """Fixture for creating a KC3DDataset instance with parameterized split."""
    split = request.param
    return KC3DDataset(data_root=kc_3d_data_root, split=split)
