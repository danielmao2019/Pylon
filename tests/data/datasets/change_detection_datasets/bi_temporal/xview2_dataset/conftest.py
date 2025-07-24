"""Shared fixtures and helper functions for xView2Dataset tests."""

import pytest
import os


@pytest.fixture
def create_dummy_xview2_files():
    """Fixture that returns a function to create minimal xView2 dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create minimal xView2 dataset structure for testing."""
        for split in ['train', 'test', 'hold']:
            split_dir = os.path.join(temp_dir, split)
            
            # Create images directory
            images_dir = os.path.join(split_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            # Create labels directory
            labels_dir = os.path.join(split_dir, 'labels')
            os.makedirs(labels_dir, exist_ok=True)
            
            # Create targets directory
            targets_dir = os.path.join(split_dir, 'targets')
            os.makedirs(targets_dir, exist_ok=True)
            
            # Create dummy files
            test_id = 'test_00000001'
            
            # Pre-disaster image
            pre_img_path = os.path.join(images_dir, f'{test_id}_pre_disaster.png')
            with open(pre_img_path, 'wb') as f:
                f.write(b'dummy_pre_image_data')
                
            # Post-disaster image
            post_img_path = os.path.join(images_dir, f'{test_id}_post_disaster.png')
            with open(post_img_path, 'wb') as f:
                f.write(b'dummy_post_image_data')
            
            # Pre-disaster label
            pre_lbl_path = os.path.join(labels_dir, f'{test_id}_pre_disaster.json')
            with open(pre_lbl_path, 'w') as f:
                f.write('{"features": []}')
                
            # Post-disaster label
            post_lbl_path = os.path.join(labels_dir, f'{test_id}_post_disaster.json')
            with open(post_lbl_path, 'w') as f:
                f.write('{"features": []}')
            
            # Targets (masks)
            pre_mask_path = os.path.join(targets_dir, f'{test_id}_pre_disaster_target.png')
            with open(pre_mask_path, 'wb') as f:
                f.write(b'dummy_pre_mask_data')
                
            post_mask_path = os.path.join(targets_dir, f'{test_id}_post_disaster_target.png')
            with open(post_mask_path, 'wb') as f:
                f.write(b'dummy_post_mask_data')
    
    return _create_files