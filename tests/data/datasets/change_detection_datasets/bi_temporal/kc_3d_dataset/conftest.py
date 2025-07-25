"""Shared fixtures and helper functions for KC3DDataset tests."""

import pytest
import os
import pickle
import numpy as np
from PIL import Image


@pytest.fixture
def create_dummy_kc3d_files():
    """Fixture that returns a function to create minimal KC3D dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create minimal KC3D dataset structure for testing."""
        # Create data split file
        data_split = {
            'train': [
                {
                    'image1': 'scene1_frame1.png',
                    'image2': 'scene1_frame2.png',
                    'mask1': 'scene1_mask1.png',
                    'mask2': 'scene1_mask2.png',
                    'depth1': 'scene1_depth1.png',
                    'depth2': 'scene1_depth2.png',
                }
            ],
            'val': [
                {
                    'image1': 'scene2_frame1.png',
                    'image2': 'scene2_frame2.png',
                    'mask1': 'scene2_mask1.png',
                    'mask2': 'scene2_mask2.png',
                    'depth1': 'scene2_depth1.png',
                    'depth2': 'scene2_depth2.png',
                }
            ],
            'test': [
                {
                    'image1': 'scene3_frame1.png',
                    'image2': 'scene3_frame2.png',
                    'mask1': 'scene3_mask1.png',
                    'mask2': 'scene3_mask2.png',
                    'depth1': 'scene3_depth1.png',
                    'depth2': 'scene3_depth2.png',
                }
            ]
        }
        
        with open(os.path.join(temp_dir, 'data_split.pkl'), 'wb') as f:
            pickle.dump(data_split, f)
        
        # Create dummy files for all splits
        for split_data in data_split.values():
            for item in split_data:
                for key, filename in item.items():
                    filepath = os.path.join(temp_dir, filename)
                    # Create 4-channel RGBA images for image files
                    if 'image' in key:
                        dummy_data = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
                    else:
                        dummy_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
                    
                    if 'image' in key:
                        img = Image.fromarray(dummy_data, 'RGBA')
                    else:
                        img = Image.fromarray(dummy_data, 'L')
                    img.save(filepath)
        
        # Create metadata files for ground truth registration
        for scene in ['scene1', 'scene2', 'scene3']:
            metadata = {
                'intrinsics': np.eye(3).tolist(),
                'position_before': [0, 0, 0],
                'position_after': [1, 1, 1],
                'rotation_before': np.eye(3).tolist(),
                'rotation_after': np.eye(3).tolist(),
            }
            np.save(os.path.join(temp_dir, f'{scene}_frame.npy'), metadata)
    
    return _create_files
