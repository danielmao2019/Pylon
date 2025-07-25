"""Tests for BaseDataset cache version discrimination."""

import pytest
import tempfile


def test_base_dataset_version_discrimination(mock_dataset_class):
    """Test that BaseDataset instances with different configurations have different version hashes."""
    with tempfile.TemporaryDirectory() as temp_dir1, tempfile.TemporaryDirectory() as temp_dir2:
        # Same configuration should have same hash
        dataset1a = mock_dataset_class(data_root=temp_dir1, split='train')
        dataset1b = mock_dataset_class(data_root=temp_dir1, split='train')
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different data_root should have different hash
        dataset2 = mock_dataset_class(data_root=temp_dir2, split='train')
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different split should have different hash
        dataset3 = mock_dataset_class(data_root=temp_dir1, split='val')
        assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()
        
        # Different split_percentages should have different hash
        dataset4 = mock_dataset_class(data_root=temp_dir1, split=(0.7, 0.2, 0.1))
        dataset5 = mock_dataset_class(data_root=temp_dir1, split=(0.8, 0.1, 0.1))
        assert dataset4.get_cache_version_hash() != dataset5.get_cache_version_hash()




def test_comprehensive_version_discrimination(mock_dataset_class):
    """Comprehensive test ensuring no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir1, tempfile.TemporaryDirectory() as temp_dir2:
        datasets = []
        
        # Test different BaseDataset configurations
        for data_root in [temp_dir1, temp_dir2]:
            for split in ['train', 'val', 'test']:
                datasets.append(mock_dataset_class(data_root=data_root, split=split))
        
        # Test different split percentages
        for split_percentages in [(0.7, 0.2, 0.1), (0.8, 0.1, 0.1), (0.6, 0.3, 0.1)]:
            datasets.append(mock_dataset_class(data_root=temp_dir1, split=split_percentages))
        
        # Collect all hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        
        # Ensure all hashes are unique (no collisions)
        assert len(hashes) == len(set(hashes)), f"Hash collision detected! Duplicate hashes found in: {hashes}"
        
        # Ensure all hashes are non-empty strings
        for hash_val in hashes:
            assert isinstance(hash_val, str), f"Hash must be string, got {type(hash_val)}"
            assert len(hash_val) > 0, "Hash must not be empty"
            assert len(hash_val) == 16, f"Hash must be 16 characters, got {len(hash_val)}"


