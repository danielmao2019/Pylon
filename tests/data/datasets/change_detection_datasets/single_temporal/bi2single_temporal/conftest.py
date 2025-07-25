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
            
            # Create dummy files
            for filename in ['test_1.png']:
                for subdir in ['A', 'B', 'label']:
                    with open(os.path.join(split_dir, subdir, filename), 'w') as f:
                        f.write('dummy')
    
    return _create_files