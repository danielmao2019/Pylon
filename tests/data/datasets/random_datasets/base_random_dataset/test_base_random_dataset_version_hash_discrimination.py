"""Tests for BaseRandomDataset cache version discrimination."""

import pytest
import tempfile
import torch
from data.datasets.random_datasets.base_random_dataset import BaseRandomDataset


def test_base_random_dataset_version_discrimination():
    """Test that BaseRandomDataset instances with different parameters have different hashes."""
    
    gen_func_config1 = {
        'inputs': {
            'data': (torch.randn, {'size': (3, 224, 224)})
        },
        'labels': {
            'class': (torch.randint, {'low': 0, 'high': 10, 'size': (1,)})
        }
    }
    
    gen_func_config2 = {
        'inputs': {
            'data': (torch.randn, {'size': (3, 256, 256)})  # Different size
        },
        'labels': {
            'class': (torch.randint, {'low': 0, 'high': 10, 'size': (1,)})
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Same parameters should have same hash
        dataset1a = BaseRandomDataset(
            data_root=temp_dir,
            num_examples=100,
            gen_func_config=gen_func_config1,
            base_seed=42,
            split='all'
        )
        dataset1b = BaseRandomDataset(
            data_root=temp_dir,
            num_examples=100,
            gen_func_config=gen_func_config1,
            base_seed=42,
            split='all'
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different num_examples should have different hash
        dataset2 = BaseRandomDataset(
            data_root=temp_dir,
            num_examples=200,  # Different
            gen_func_config=gen_func_config1,
            base_seed=42,
            split='all'
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different gen_func_config should have different hash
        dataset3 = BaseRandomDataset(
            data_root=temp_dir,
            num_examples=100,
            gen_func_config=gen_func_config2,  # Different
            base_seed=42,
            split='all'
        )
        assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()
        
        # Different initial_seed should have different hash
        dataset4 = BaseRandomDataset(
            data_root=temp_dir,
            num_examples=100,
            gen_func_config=gen_func_config1,
            base_seed=123,  # Different
            split='all'
        )
        assert dataset1a.get_cache_version_hash() != dataset4.get_cache_version_hash()


def test_all_parameters_affect_version_hash():
    """Test that all content-affecting parameters impact the version hash."""
    
    gen_func_config_variants = [
        {
            'inputs': {
                'data': (torch.randn, {'size': (3, 224, 224)})
            },
            'labels': {
                'class': (torch.randint, {'low': 0, 'high': 10, 'size': (1,)})
            }
        },
        {
            'inputs': {
                'data': (torch.randn, {'size': (3, 256, 256)})  # Different size
            },
            'labels': {
                'class': (torch.randint, {'low': 0, 'high': 10, 'size': (1,)})
            }
        },
        {
            'inputs': {
                'data': (torch.randn, {'size': (3, 224, 224)})
            },
            'labels': {
                'class': (torch.randint, {'low': 0, 'high': 20, 'size': (1,)})  # Different high
            }
        }
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_args = {
            'data_root': temp_dir,
            'num_examples': 100,
            'gen_func_config': gen_func_config_variants[0],
            'base_seed': 42,
            'split': 'all',
        }
        
        # Test each parameter individually
        parameter_variants = [
            ('num_examples', 200),
            ('gen_func_config', gen_func_config_variants[1]),
            ('gen_func_config', gen_func_config_variants[2]),
            ('initial_seed', 123),
            ('initial_seed', None),  # None vs specified
        ]
        
        dataset1 = BaseRandomDataset(**base_args)
        
        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = BaseRandomDataset(**modified_args)
            
            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Parameter {param_name} should affect cache version hash"


def test_none_vs_specified_initial_seed():
    """Test that None vs specified initial_seed produces different hashes."""
    
    gen_func_config = {
        'inputs': {
            'data': (torch.randn, {'size': (3, 224, 224)})
        },
        'labels': {
            'class': (torch.randint, {'low': 0, 'high': 10, 'size': (1,)})
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # None vs specified should have different hashes
        dataset1 = BaseRandomDataset(
            data_root=temp_dir,
            num_examples=100,
            gen_func_config=gen_func_config,
            base_seed=None,
            split='all'
        )
        dataset2 = BaseRandomDataset(
            data_root=temp_dir,
            num_examples=100,
            gen_func_config=gen_func_config,
            base_seed=42,
            split='all'
        )
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_gen_func_config_serialization():
    """Test that different gen_func_config structures are properly discriminated."""
    
    config_variants = [
        # Same structure, different parameters
        {
            'inputs': {'data': (torch.randn, {'size': (3, 224, 224)})},
            'labels': {'class': (torch.randint, {'low': 0, 'high': 10, 'size': (1,)})}
        },
        {
            'inputs': {'data': (torch.randn, {'size': (3, 256, 256)})},  # Different size
            'labels': {'class': (torch.randint, {'low': 0, 'high': 10, 'size': (1,)})}
        },
        # Different structure (additional input)
        {
            'inputs': {
                'data': (torch.randn, {'size': (3, 224, 224)}),
                'mask': (torch.randint, {'low': 0, 'high': 2, 'size': (224, 224)})  # Additional input
            },
            'labels': {'class': (torch.randint, {'low': 0, 'high': 10, 'size': (1,)})}
        },
        # Different function
        {
            'inputs': {'data': (torch.zeros, {'size': (3, 224, 224)})},  # Different function
            'labels': {'class': (torch.randint, {'low': 0, 'high': 10, 'size': (1,)})}
        }
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        datasets = []
        for config in config_variants:
            dataset = BaseRandomDataset(
                data_root=temp_dir,
                num_examples=100,
                gen_func_config=config,
                base_seed=42,
                split='all'
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All gen_func_config variants should produce different hashes, got: {hashes}"


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    
    gen_func_configs = [
        {
            'inputs': {'data': (torch.randn, {'size': (3, 224, 224)})},
            'labels': {'class': (torch.randint, {'low': 0, 'high': 10, 'size': (1,)})}
        },
        {
            'inputs': {'data': (torch.randn, {'size': (3, 256, 256)})},
            'labels': {'class': (torch.randint, {'low': 0, 'high': 10, 'size': (1,)})}
        },
        {
            'inputs': {'data': (torch.randn, {'size': (1, 128, 128)})},
            'labels': {'class': (torch.randint, {'low': 0, 'high': 5, 'size': (1,)})}
        }
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        datasets = []
        
        # Generate many different dataset configurations
        for num_examples in [50, 100, 200]:
            for config in gen_func_configs:
                for base_seed_val in [None, 42, 123]:
                    datasets.append(BaseRandomDataset(
                        data_root=temp_dir,
                        num_examples=num_examples,
                        gen_func_config=config,
                        base_seed=base_seed_val,
                        split='all'
                    ))
        
        # Collect all hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        
        # Ensure all hashes are unique (no collisions)
        assert len(hashes) == len(set(hashes)), \
            f"Hash collision detected! Duplicate hashes found in: {hashes}"
        
        # Ensure all hashes are properly formatted
        for hash_val in hashes:
            assert isinstance(hash_val, str), f"Hash must be string, got {type(hash_val)}"
            assert len(hash_val) == 16, f"Hash must be 16 characters, got {len(hash_val)}"


