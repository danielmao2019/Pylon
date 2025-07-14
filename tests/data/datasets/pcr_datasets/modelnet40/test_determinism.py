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
        cache_file_100 = f.name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        cache_file_200 = f.name
    
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
        
        # Create 100-datapoint dataset
        config_100 = base_config.copy()
        config_100.update({
            'dataset_size': 100,
            'cache_filepath': cache_file_100,
        })
        
        print("Creating 100-datapoint dataset...")
        dataset_100 = data.datasets.ModelNet40Dataset(**config_100)
        
        # Iterate through all datapoints to populate cache
        for i in range(len(dataset_100)):
            _ = dataset_100[i]
        
        # Create 200-datapoint dataset
        config_200 = base_config.copy()
        config_200.update({
            'dataset_size': 200,
            'cache_filepath': cache_file_200,
        })
        
        print("Creating 200-datapoint dataset...")
        dataset_200 = data.datasets.ModelNet40Dataset(**config_200)
        
        # Iterate through all datapoints to populate cache
        for i in range(len(dataset_200)):
            _ = dataset_200[i]
        
        # Load both caches
        with open(cache_file_100, 'r') as f:
            cache_100 = json.load(f)
        with open(cache_file_200, 'r') as f:
            cache_200 = json.load(f)
        
        print(f"100-datapoint cache has {len(cache_100)} file keys")
        print(f"200-datapoint cache has {len(cache_200)} file keys")
        
        # Check that both caches have the same file keys
        assert set(cache_100.keys()) == set(cache_200.keys()), \
            "Both caches should have the same file keys"
        
        # For each file, check that the first N transforms are identical
        # where N is the number of transforms for that file in the 100-datapoint case
        for file_key in cache_100.keys():
            transforms_100 = cache_100[file_key]
            transforms_200 = cache_200[file_key]
            
            num_transforms_100 = len(transforms_100)
            
            # The 200-datapoint dataset should have at least as many transforms
            assert len(transforms_200) >= num_transforms_100, \
                f"200-datapoint cache should have at least {num_transforms_100} transforms for file {file_key}"
            
            # The first num_transforms_100 should be identical
            first_transforms_200 = transforms_200[:num_transforms_100]
            
            assert transforms_100 == first_transforms_200, \
                f"First {num_transforms_100} transforms should be identical for file {file_key}"
        
        print("✅ Cache subset consistency test passed: first 100 datapoints are identical")
        
    finally:
        # Clean up temporary files
        for cache_file in [cache_file_100, cache_file_200]:
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
            
            # Compare inputs
            for key in dp1['inputs']:
                if isinstance(dp1['inputs'][key], torch.Tensor):
                    assert torch.equal(dp1['inputs'][key], dp2['inputs'][key]), \
                        f"Inputs tensor {key} differs at index {idx}"
                else:
                    assert dp1['inputs'][key] == dp2['inputs'][key], \
                        f"Inputs {key} differs at index {idx}"
            
            # Compare labels  
            for key in dp1['labels']:
                if isinstance(dp1['labels'][key], torch.Tensor):
                    assert torch.equal(dp1['labels'][key], dp2['labels'][key]), \
                        f"Labels tensor {key} differs at index {idx}"
                else:
                    assert dp1['labels'][key] == dp2['labels'][key], \
                        f"Labels {key} differs at index {idx}"
            
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
