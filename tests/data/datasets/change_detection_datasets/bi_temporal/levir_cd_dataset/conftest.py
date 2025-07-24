"""Shared fixtures and helper functions for LevirCdDataset tests."""

import pytest
import os
import numpy as np
from PIL import Image


@pytest.fixture
def create_dummy_levir_structure():
    """Fixture that returns a function to create a dummy LEVIR-CD directory structure for testing."""
    def _create_structure(data_root: str) -> None:
        """Create a dummy LEVIR-CD directory structure for testing."""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            # Create directories
            split_dir = os.path.join(data_root, split)
            a_dir = os.path.join(split_dir, 'A')
            b_dir = os.path.join(split_dir, 'B')
            label_dir = os.path.join(split_dir, 'label')
            
            os.makedirs(a_dir, exist_ok=True)
            os.makedirs(b_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            
            # Create dummy image files
            num_files = 3 if split == 'train' else 2
            for i in range(num_files):
                filename = f'test_{i:04d}.png'
                
                # Create dummy images (RGB for inputs, grayscale for labels)
                img_a = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
                img_b = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
                label = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8))
                
                img_a.save(os.path.join(a_dir, filename))
                img_b.save(os.path.join(b_dir, filename))
                label.save(os.path.join(label_dir, filename))
    
    return _create_structure