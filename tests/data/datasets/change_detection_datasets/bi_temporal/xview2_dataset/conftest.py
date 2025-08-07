"""Shared fixtures and helper functions for xView2 dataset tests."""

import pytest
import os
import tempfile
import numpy as np
from contextlib import contextmanager
from PIL import Image
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset


@pytest.fixture
def xview2_dataset_train_config(xview2_temp_data_root):
    """Fixture for creating an xView2Dataset config with train split."""
    return {
        'class': xView2Dataset,
        'args': {
            'data_root': xview2_temp_data_root,
            'split': 'train'
        }
    }


@pytest.fixture  
def create_dummy_xview2_data():
    """Fixture that creates dummy xView2 dataset structure."""
    def _create_dummy_data(data_root: str) -> None:
        """Create dummy xView2 dataset files."""
        # Create directory structure for all splits
        splits_structure = {
            'tier1': 5,    # 5 samples for tier1 (part of train)
            'tier3': 3,    # 3 samples for tier3 (part of train) 
            'test': 4,     # 4 samples for test
            'hold': 4,     # 4 samples for hold
        }
        
        # Create dummy RGB image (3 channels, 256x256)
        dummy_rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Create dummy label image (single channel, values 0-4)
        dummy_label = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
        
        for split, num_samples in splits_structure.items():
            # Create images and targets directories
            images_dir = os.path.join(data_root, split, 'images')
            targets_dir = os.path.join(data_root, split, 'targets')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(targets_dir, exist_ok=True)
            
            for i in range(num_samples):
                sample_id = f"sample_{i:04d}"
                
                # Create pre and post disaster images
                pre_img_path = os.path.join(images_dir, f"{sample_id}_pre_disaster.png")
                post_img_path = os.path.join(images_dir, f"{sample_id}_post_disaster.png")
                Image.fromarray(dummy_rgb, mode='RGB').save(pre_img_path)
                Image.fromarray(dummy_rgb, mode='RGB').save(post_img_path)
                
                # Create corresponding label files (note the _target suffix)
                pre_lbl_path = os.path.join(targets_dir, f"{sample_id}_pre_disaster_target.png")
                post_lbl_path = os.path.join(targets_dir, f"{sample_id}_post_disaster_target.png")
                Image.fromarray(dummy_label, mode='L').save(pre_lbl_path)
                Image.fromarray(dummy_label, mode='L').save(post_lbl_path)
    
    return _create_dummy_data


@pytest.fixture
def patched_xview2_dataset_size():
    """Fixture that patches xView2 DATASET_SIZE for testing."""
    @contextmanager
    def _patch():
        """Context manager to temporarily patch DATASET_SIZE for testing."""
        original_dataset_size = xView2Dataset.DATASET_SIZE
        # Update sizes to match our dummy data
        xView2Dataset.DATASET_SIZE = {
            'train': 8,    # tier1 (5) + tier3 (3) = 8 samples  
            'test': 4,     # 4 samples for test
            'hold': 4,     # 4 samples for hold
        }
        try:
            yield
        finally:
            xView2Dataset.DATASET_SIZE = original_dataset_size
    
    return _patch


@pytest.fixture
def xview2_temp_data_root(create_dummy_xview2_data):
    """Fixture that creates a temporary directory with xView2 dummy data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy data structure
        create_dummy_xview2_data(temp_dir)
        yield temp_dir


@pytest.fixture
def dataset_config(request, xview2_temp_data_root):
    """Fixture for creating an xView2Dataset config with parameterized split."""
    split = request.param
    return {
        'class': xView2Dataset,
        'args': {
            'data_root': xview2_temp_data_root,
            'split': split
        }
    }
