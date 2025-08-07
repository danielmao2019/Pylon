"""Test cache version discrimination for OSCDDataset."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset
from utils.builders.builder import build_from_config
import copy


def test_oscd_hash_consistency(oscd_dataset_train_config, oscd_dataset_test_config):
    """Test that OSCDDataset produces different hashes for different parameters."""
    
    oscd_dataset_train = build_from_config(oscd_dataset_train_config)
    oscd_dataset_test = build_from_config(oscd_dataset_test_config)
    
    # Same parameters should produce same hash - create another instance with same params
    dataset_train_copy = build_from_config(oscd_dataset_train_config)
    assert oscd_dataset_train.get_cache_version_hash() == dataset_train_copy.get_cache_version_hash()
    
    # Different split should produce different hash
    assert oscd_dataset_train.get_cache_version_hash() != oscd_dataset_test.get_cache_version_hash()


def test_oscd_unique_hashes(oscd_dataset_train_config, oscd_dataset_test_config):
    """Test that different OSCD configurations produce unique hashes."""
    
    # Build datasets from configs
    oscd_dataset_train = build_from_config(oscd_dataset_train_config)
    oscd_dataset_test = build_from_config(oscd_dataset_test_config)
    
    # Collect hashes from different configurations
    datasets = [oscd_dataset_train, oscd_dataset_test]
    hashes = []
    
    for dataset in datasets:
        hash_val = dataset.get_cache_version_hash()
        assert hash_val not in hashes, f"Hash collision detected for dataset {dataset}"
        hashes.append(hash_val)
    
    assert len(hashes) == len(datasets), "Some datasets produced duplicate hashes"
    print(f"Generated {len(hashes)} unique hashes for OSCD dataset configurations")
