"""Shared fixtures and helper functions for AirChangeDataset tests."""

import pytest
import os


@pytest.fixture
def create_dummy_air_change_files():
    """Fixture that returns a function to create minimal Air Change dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create minimal Air Change dataset structure for testing."""
        folders = ["Szada", "Tiszadob"]
        
        for folder in folders:
            folder_path = os.path.join(temp_dir, folder)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create dummy files
            for filename in ['test_im1.bmp', 'test_im2.bmp', 'test_gt.bmp']:
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'wb') as f:
                    f.write(b'dummy_binary_data')
    
    return _create_files