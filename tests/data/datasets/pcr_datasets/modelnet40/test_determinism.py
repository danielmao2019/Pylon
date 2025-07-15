"""
Determinism tests for ModelNet40Dataset.

These tests verify that the cache generation and datapoint loading are deterministic
and consistent between runs, ensuring reproducible results.
"""

import os
import json
import tempfile
import random
import torch
import data
from utils.ops.dict_as_tensor import buffer_allclose


def test_cache_consistency():
    """Test that cache generation is absolutely consistent between runs."""
    # Create temporary cache files
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        cache_file_1 = f.name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        cache_file_2 = f.name
    
    try:
        # Test parameters
        dataset_config = {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'train',
            'dataset_size': 20,  # Small size for faster testing
            'rotation_mag': 45.0,
            'translation_mag': 0.5,
            'matching_radius': 0.05,
            'overlap_range': (0.3, 1.0),
            'min_points': 512,
        }
        
        # First run: generate cache
        print(f"First run with cache file: {cache_file_1}")
        dataset_config_1 = dataset_config.copy()
        dataset_config_1['cache_filepath'] = cache_file_1
        
        dataset1 = data.datasets.ModelNet40Dataset(**dataset_config_1)
        
        # Iterate through all datapoints to populate cache
        for i in range(len(dataset1)):
            _ = dataset1[i]
        
        # Verify cache file was created and has content
        assert os.path.exists(cache_file_1), "Cache file should be created"
        with open(cache_file_1, 'r') as f:
            cache_content_1 = json.load(f)
        assert len(cache_content_1) > 0, "Cache should have content after iteration"
        print(f"First run cache keys: {list(cache_content_1.keys())}")
        
        # Second run: generate cache with same config
        print(f"Second run with cache file: {cache_file_2}")
        dataset_config_2 = dataset_config.copy()
        dataset_config_2['cache_filepath'] = cache_file_2
        
        dataset2 = data.datasets.ModelNet40Dataset(**dataset_config_2)
        
        # Iterate through all datapoints to populate cache
        for i in range(len(dataset2)):
            _ = dataset2[i]
        
        # Load second cache
        assert os.path.exists(cache_file_2), "Second cache file should be created"
        with open(cache_file_2, 'r') as f:
            cache_content_2 = json.load(f)
        assert len(cache_content_2) > 0, "Second cache should have content"
        print(f"Second run cache keys: {list(cache_content_2.keys())}")
        
        # Compare cache files for absolute identity
        assert cache_content_1 == cache_content_2, \
            "Cache files should be absolutely identical between runs"
        
        # Also compare as JSON strings to ensure formatting is identical
        with open(cache_file_1, 'r') as f1, open(cache_file_2, 'r') as f2:
            content1_str = f1.read()
            content2_str = f2.read()
        
        # The JSON content should be functionally identical even if formatting differs
        cache1_normalized = json.loads(content1_str)
        cache2_normalized = json.loads(content2_str)
        assert cache1_normalized == cache2_normalized, \
            "Normalized cache content should be identical"
        
        print("✅ Cache consistency test passed: both runs produced identical caches")
        
    finally:
        # Clean up temporary files
        for cache_file in [cache_file_1, cache_file_2]:
            if os.path.exists(cache_file):
                os.unlink(cache_file)


def test_cache_subset_consistency():
    """Test that first N datapoints are identical between different dataset sizes."""
    # Create temporary cache files
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        cache_file_10 = f.name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        cache_file_20 = f.name
    
    try:
        base_config = {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'train',
            'rotation_mag': 45.0,
            'translation_mag': 0.5,
            'matching_radius': 0.05,
            'overlap_range': (0.3, 1.0),
            'min_points': 512,
        }
        
        # Create 10-datapoint dataset
        config_10 = base_config.copy()
        config_10.update({
            'dataset_size': 10,
            'cache_filepath': cache_file_10,
        })
        
        print("Creating 10-datapoint dataset...")
        dataset_10 = data.datasets.ModelNet40Dataset(**config_10)
        
        # Iterate through all datapoints to populate cache
        for i in range(len(dataset_10)):
            _ = dataset_10[i]
        
        # Create 20-datapoint dataset
        config_20 = base_config.copy()
        config_20.update({
            'dataset_size': 20,
            'cache_filepath': cache_file_20,
        })
        
        print("Creating 20-datapoint dataset...")
        dataset_20 = data.datasets.ModelNet40Dataset(**config_20)
        
        # Iterate through all datapoints to populate cache
        for i in range(len(dataset_20)):
            _ = dataset_20[i]
        
        # Load both caches
        with open(cache_file_10, 'r') as f:
            cache_10 = json.load(f)
        with open(cache_file_20, 'r') as f:
            cache_20 = json.load(f)
        
        print(f"10-datapoint cache has {len(cache_10)} file keys")
        print(f"20-datapoint cache has {len(cache_20)} file keys")
        
        # Check that cache_10 keys are a subset of cache_20 keys
        # (20-datapoint dataset should include all files from 10-datapoint dataset)
        cache_10_keys = set(cache_10.keys())
        cache_20_keys = set(cache_20.keys())
        
        assert cache_10_keys.issubset(cache_20_keys), \
            f"10-datapoint cache keys should be subset of 20-datapoint cache keys. " \
            f"Missing: {cache_10_keys - cache_20_keys}"
        
        # For each file, check that the first N transforms are identical
        # where N is the number of transforms for that file in the 10-datapoint case
        for file_key in cache_10.keys():
            transforms_10 = cache_10[file_key]
            transforms_20 = cache_20[file_key]
            
            num_transforms_10 = len(transforms_10)
            
            # The 20-datapoint dataset should have at least as many transforms
            assert len(transforms_20) >= num_transforms_10, \
                f"20-datapoint cache should have at least {num_transforms_10} transforms for file {file_key}"
            
            # The first num_transforms_10 should be identical
            first_transforms_20 = transforms_20[:num_transforms_10]
            
            assert transforms_10 == first_transforms_20, \
                f"First {num_transforms_10} transforms should be identical for file {file_key}"
        
        print("✅ Cache subset consistency test passed: first 10 datapoints are identical")
        
    finally:
        # Clean up temporary files
        for cache_file in [cache_file_10, cache_file_20]:
            if os.path.exists(cache_file):
                os.unlink(cache_file)


def test_datapoint_determinism():
    """Test that same datapoints are identical across dataset instances."""
    # Create temporary cache file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        cache_file = f.name
    
    try:
        dataset_config = {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'train', 
            'dataset_size': 50,
            'rotation_mag': 45.0,
            'translation_mag': 0.5,
            'matching_radius': 0.05,
            'overlap_range': (0.3, 1.0),
            'min_points': 512,
            'cache_filepath': cache_file,
        }
        
        # Create first dataset instance
        print("Creating first dataset instance...")
        dataset1 = data.datasets.ModelNet40Dataset(**dataset_config)
        
        # Randomly sample 10 indices to test
        test_indices = random.sample(range(len(dataset1)), min(10, len(dataset1)))
        print(f"Testing indices: {test_indices}")
        
        # Load datapoints from first instance
        datapoints1 = {}
        for idx in test_indices:
            datapoint = dataset1[idx]
            # Deep copy the datapoint to avoid reference issues
            datapoints1[idx] = {
                'inputs': {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in datapoint['inputs'].items()},
                'labels': {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in datapoint['labels'].items()},
                'meta_info': datapoint['meta_info'].copy()
            }
        
        # Create second dataset instance with same config
        print("Creating second dataset instance...")
        dataset2 = data.datasets.ModelNet40Dataset(**dataset_config)
        
        # Load same datapoints from second instance
        datapoints2 = {}
        for idx in test_indices:
            datapoint = dataset2[idx]
            datapoints2[idx] = {
                'inputs': {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in datapoint['inputs'].items()},
                'labels': {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in datapoint['labels'].items()},
                'meta_info': datapoint['meta_info'].copy()
            }
        
        # Compare datapoints for exact equality
        for idx in test_indices:
            dp1 = datapoints1[idx]
            dp2 = datapoints2[idx]
            
            # Compare inputs using buffer_allclose for tensors
            assert buffer_allclose(dp1['inputs'], dp2['inputs']), \
                f"Inputs differ at index {idx}"
            
            # Compare labels using buffer_allclose for tensors
            assert buffer_allclose(dp1['labels'], dp2['labels']), \
                f"Labels differ at index {idx}"
            
            # Compare meta_info (skip dynamic fields like timestamps if any)
            meta_keys_to_compare = ['file_idx', 'transform_idx', 'overlap', 'crop_method', 'keep_ratio']
            for key in meta_keys_to_compare:
                if key in dp1['meta_info'] and key in dp2['meta_info']:
                    assert dp1['meta_info'][key] == dp2['meta_info'][key], \
                        f"Meta_info {key} differs at index {idx}: {dp1['meta_info'][key]} vs {dp2['meta_info'][key]}"
        
        print("✅ Datapoint determinism test passed: same indices produce identical datapoints")
        
    finally:
        # Clean up temporary files
        if os.path.exists(cache_file):
            os.unlink(cache_file)
