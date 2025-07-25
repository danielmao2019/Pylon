"""Shared fixtures and helper functions for cdd_dataset tests."""

import pytest
import os


@pytest.fixture
def create_dummy_cdd_files():
    """Fixture that returns a function for creating dummy dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create minimal CDD dataset structure for testing."""
        for split in ['train', 'val', 'test']:
            # Create structure with both regular and with_shift folders
            for folder_type in ['regular', 'with_shift']:
                if folder_type == 'regular':
                    base_path = os.path.join(temp_dir, 'folder1', 'subfolder1', split)
                else:
                    base_path = os.path.join(temp_dir, 'folder1', 'with_shift', split)
                
                for subdir in ['A', 'B', 'OUT']:
                    os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
                    
                    # Create dummy files
                    file_ext = '.jpg' if split == 'train' else '.bmp'
                    if folder_type == 'with_shift' and split != 'train':
                        file_ext = '.bmp'
                    elif folder_type == 'regular':
                        file_ext = '.jpg'
                    
                    for filename in [f'test_1{file_ext}']:
                        filepath = os.path.join(base_path, subdir, filename)
                        with open(filepath, 'wb') as f:
                            f.write(b'dummy_image_data')


    
    return _create_files
