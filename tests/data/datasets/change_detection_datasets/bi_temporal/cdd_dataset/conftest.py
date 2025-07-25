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
                    
                    # Create efficient dummy files using minimal I/O
                    # Split total size between regular and with_shift folders
                    dataset_sizes = {'train': 26000, 'val': 6998, 'test': 7000}
                    total_files = dataset_sizes[split]
                    
                    # Split files between regular and with_shift (roughly half each)
                    if folder_type == 'regular':
                        num_files = total_files // 2
                    else:  # with_shift
                        num_files = total_files - (total_files // 2)  # Remainder goes to with_shift
                    
                    file_ext = '.jpg' if split == 'train' else '.bmp'
                    if folder_type == 'with_shift' and split != 'train':
                        file_ext = '.bmp'
                    elif folder_type == 'regular':
                        file_ext = '.jpg'
                    
                    # Create files efficiently using batch operations
                    import subprocess
                    filenames = [f'test_{i:06d}{file_ext}' for i in range(num_files)]
                    
                    # Use touch command for fast file creation (much faster than Python loops)
                    subprocess.run(['touch'] + [os.path.join(base_path, subdir, f) for f in filenames], check=True)


    
    return _create_files
