"""Shared fixtures and helper functions for slpccd_dataset tests."""

import pytest
import os


@pytest.fixture
def create_dummy_slpccd_files():
    """Fixture that returns a function for creating dummy dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create minimal SLPCCD dataset structure for testing."""
        # Create split files
        for split in ['train', 'val', 'test']:
            split_file = os.path.join(temp_dir, f'{split}.txt')
            with open(split_file, 'w') as f:
                f.write('test_pair_1\n')
        
        # Create dummy point cloud files
        test_pair_dir = os.path.join(temp_dir, 'test_pair_1')
        os.makedirs(test_pair_dir, exist_ok=True)
        
        # Create 2016 and 2020 point cloud files
        for year in ['2016', '2020']:
            pc_dir = os.path.join(test_pair_dir, year)
            os.makedirs(pc_dir, exist_ok=True)
            
            # Create point cloud file
            pc_file = os.path.join(pc_dir, 'points.txt')
            with open(pc_file, 'w') as f:
                # Write dummy point cloud data (x, y, z, r, g, b)
                for i in range(100):
                    f.write(f'{i/10.0} {i/20.0} {i/30.0} 255 255 255\n')
        
        # Create change labels file
        labels_file = os.path.join(test_pair_dir, 'change_labels.txt')
        with open(labels_file, 'w') as f:
            # Write dummy change labels (0 for unchanged, 1 for changed)
            for i in range(100):
                f.write(f'{i % 2}\n')
    
    return _create_files