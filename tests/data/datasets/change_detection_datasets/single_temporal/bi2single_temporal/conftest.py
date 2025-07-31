"""Shared fixtures and helper functions for Bi2SingleTemporal tests."""

import pytest
import os


@pytest.fixture
def create_dummy_levir_cd_files():
    """Fixture that returns a function to create minimal LEVIR-CD dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create minimal LEVIR-CD dataset structure for testing."""
        for split in ['train', 'test']:
            split_dir = os.path.join(temp_dir, split)
            os.makedirs(os.path.join(split_dir, 'A'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'B'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'label'), exist_ok=True)
            
            # Create dummy files efficiently using touch command
            dataset_sizes = {'train': 445, 'test': 128}
            num_files = dataset_sizes.get(split, 1)  # Default to 1 if split not recognized
            
            # Generate all filenames
            filenames = [f'test_{i:04d}.png' for i in range(num_files)]
            
            # Use touch command for fast file creation (much faster than Python loops)
            import subprocess
            for subdir in ['A', 'B', 'label']:
                full_paths = [os.path.join(split_dir, subdir, f) for f in filenames]
                subprocess.run(['touch'] + full_paths, check=True)
    
    return _create_files
