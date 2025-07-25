"""Test cache version discrimination for OSCDDataset."""

import pytest
import tempfile
import os
import json
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset


# Real OSCD dataset path
OSCD_DATA_ROOT = os.environ.get('OSCD_DATA_ROOT', './data/datasets/soft_links/OSCD')


def test_oscd_dataset_version_discrimination():
    """Test that OSCDDataset produces different hashes for different parameters."""
    
    # Same parameters should produce same hash
    dataset1a = OSCDDataset(data_root=OSCD_DATA_ROOT, split='train')
    dataset1b = OSCDDataset(data_root=OSCD_DATA_ROOT, split='train')
    assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
    
    # Different split should produce different hash
    dataset2 = OSCDDataset(data_root=OSCD_DATA_ROOT, split='test')
    assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()




def test_oscd_dataset_comprehensive_no_hash_collisions():
    """Test that different OSCD configurations produce unique hashes."""
    
    # Test various parameter combinations (note: OSCD only has 'train' and 'test' splits)
    configs = [
        {'data_root': OSCD_DATA_ROOT, 'split': 'train'},
        {'data_root': OSCD_DATA_ROOT, 'split': 'test'},
    ]
    
    hashes = []
    for config in configs:
        dataset = OSCDDataset(**config)
        hash_val = dataset.get_cache_version_hash()
        assert hash_val not in hashes, f"Hash collision detected for config {config}"
        hashes.append(hash_val)
    
    print(f"Generated {len(hashes)} unique hashes for OSCD dataset configurations")
