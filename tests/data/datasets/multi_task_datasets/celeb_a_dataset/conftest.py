"""Shared fixtures and helper functions for CelebADataset tests."""

import pytest
import os
from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset


@pytest.fixture
def create_dummy_celeb_a_files():
    """Fixture that returns a function to create minimal CelebA dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create minimal CelebA dataset structure for testing."""
        images_dir = os.path.join(temp_dir, 'images', 'img_align_celeba')
        os.makedirs(images_dir, exist_ok=True)
        
        # Create dummy files
        with open(os.path.join(images_dir, '000001.jpg'), 'w') as f:
            f.write('dummy')
        with open(os.path.join(images_dir, '000002.jpg'), 'w') as f:
            f.write('dummy')
        
        # Create partition file
        with open(os.path.join(temp_dir, 'list_eval_partition.txt'), 'w') as f:
            f.write('000001.jpg 0\n000002.jpg 1\n')
        
        # Create landmarks file
        with open(os.path.join(temp_dir, 'list_landmarks_align_celeba.txt'), 'w') as f:
            f.write('lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y\n')
            f.write('000001.jpg 1 1 1 1 1 1 1 1 1 1\n')
            f.write('000002.jpg 2 2 2 2 2 2 2 2 2 2\n')
        
        # Create attributes file
        with open(os.path.join(temp_dir, 'list_attr_celeba.txt'), 'w') as f:
            attrs = CelebADataset.LABEL_NAMES[1:]  # Skip 'landmarks'
            f.write(' '.join(attrs) + '\n')
            f.write('000001.jpg ' + ' '.join(['1'] * len(attrs)) + '\n')
            f.write('000002.jpg ' + ' '.join(['-1'] * len(attrs)) + '\n')
    
    return _create_files