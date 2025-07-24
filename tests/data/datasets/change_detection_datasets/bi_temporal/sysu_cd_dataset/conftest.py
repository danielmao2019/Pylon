"""Shared fixtures and helper functions for SYSU_CD_Dataset tests."""

import pytest
import os


@pytest.fixture
def create_dummy_sysu_cd_files():
    """Fixture that returns a function to create minimal SYSU-CD dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create minimal SYSU-CD dataset structure for testing."""
        for split in ['train', 'val', 'test']:
            # Create directory structure
            time1_dir = os.path.join(temp_dir, split, 'time1')
            time2_dir = os.path.join(temp_dir, split, 'time2')
            label_dir = os.path.join(temp_dir, split, 'label')
            
            os.makedirs(time1_dir, exist_ok=True)
            os.makedirs(time2_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            
            # Create dummy files
            for filename in ['test_1.png']:
                for subdir in [time1_dir, time2_dir, label_dir]:
                    filepath = os.path.join(subdir, filename)
                    with open(filepath, 'wb') as f:
                        f.write(b'dummy_image_data')
    
    return _create_files