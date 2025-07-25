"""Test cache version discrimination for OSCDDataset."""

import pytest
import tempfile
import os
import json
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset




def test_oscd_dataset_version_discrimination(create_dummy_oscd_files):
    """Test that OSCDDataset produces different hashes for different parameters."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_oscd_files(temp_dir)
        
        # Same parameters should produce same hash
        dataset1a = OSCDDataset(data_root=temp_dir, split='train')
        dataset1b = OSCDDataset(data_root=temp_dir, split='train')
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different split should produce different hash
        dataset2 = OSCDDataset(data_root=temp_dir, split='test')
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()




def test_oscd_dataset_comprehensive_no_hash_collisions(create_dummy_oscd_files):
    """Test that different OSCD configurations produce unique hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_oscd_files(temp_dir)
        
        # Test various parameter combinations
        configs = [
            {'data_root': temp_dir, 'split': 'train'},
            {'data_root': temp_dir, 'split': 'test'},
            {'data_root': temp_dir, 'split': 'val'},
        ]
        
        hashes = []
        for config in configs:
            dataset = OSCDDataset(**config)
            hash_val = dataset.get_cache_version_hash()
            assert hash_val not in hashes, f"Hash collision detected for config {config}"
            hashes.append(hash_val)
        
        print(f"Generated {len(hashes)} unique hashes for OSCD dataset configurations")


