"""Shared fixtures and helper functions for AirChangeDataset tests."""

import pytest
import os


@pytest.fixture
def create_dummy_air_change_files():
    """Fixture that returns a function to create Air Change dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create Air Change dataset structure with 12 images (6 per folder) to produce 3744 crops."""
        folders = ["Szada", "Tiszadob"]
        
        for folder in folders:
            folder_path = os.path.join(temp_dir, folder)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create 6 image sets per folder (12 total) to match expected dataset size
            # Each image produces 312 crops (3744 / 12), so 12 images total
            for i in range(6):
                for suffix in ['im1', 'im2', 'gt']:
                    filename = f'test_{i:04d}_{suffix}.bmp'
                    filepath = os.path.join(folder_path, filename)
                    with open(filepath, 'wb') as f:
                        f.write(b'dummy_binary_data')
    
    return _create_files
