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
            
            # Create dummy image files efficiently using touch command
            dataset_sizes = {'train': 445, 'val': 64, 'test': 128}
            num_files = dataset_sizes[split]
            
            # Generate all filenames
            filenames = [f'test_{i:04d}.png' for i in range(num_files)]
            
            # Use touch command for fast file creation (much faster than Python loops)
            import subprocess
            for subdir_path in [a_dir, b_dir, label_dir]:
                full_paths = [os.path.join(subdir_path, f) for f in filenames]
                subprocess.run(['touch'] + full_paths, check=True)
    
    return _create_structure
