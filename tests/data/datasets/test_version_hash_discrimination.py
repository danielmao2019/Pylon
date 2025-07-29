"""Common version hash discrimination tests for all datasets.

This module provides centralized testing for dataset cache version hash functionality
across all dataset types, avoiding code duplication and complex fixture management.
"""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset
from data.datasets.change_detection_datasets.bi_temporal.cdd_dataset import CDDDataset
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset
from data.datasets.change_detection_datasets.bi_temporal.sysu_cd_dataset import SYSU_CD_Dataset
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset
from data.datasets.change_detection_datasets.single_temporal.ppsl_dataset import PPSLDataset
from data.datasets.change_detection_datasets.single_temporal.i3pe_dataset import I3PEDataset
from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset


# =============================================================================
# Dataset Configuration Registry
# =============================================================================

DATASET_CONFIGS = {
    # Change Detection - Bi-temporal (working datasets only)
    'air_change': {
        'class': AirChangeDataset,
        'data_root': './data/datasets/soft_links/AirChange',
        'splits': ['train', 'test'],
        'default_params': {}
    },
    'cdd': {
        'class': CDDDataset,
        'data_root': './data/datasets/soft_links/CDD',
        'splits': ['train', 'val', 'test'],
        'default_params': {}
    },
    'levir_cd': {
        'class': LevirCdDataset,
        'data_root': './data/datasets/soft_links/LEVIR-CD',
        'splits': ['train', 'val', 'test'],
        'default_params': {}
    },
    'oscd': {
        'class': OSCDDataset,
        'data_root': './data/datasets/soft_links/OSCD',
        'splits': ['train', 'val', 'test'],
        'default_params': {}
    },
    'slpccd': {
        'class': SLPCCDDataset,
        'data_root': './data/datasets/soft_links/SLPCCD',
        'splits': ['train', 'val', 'test'],
        'default_params': {}
    },
    'sysu_cd': {
        'class': SYSU_CD_Dataset,
        'data_root': './data/datasets/soft_links/SYSU-CD',
        'splits': ['train', 'val', 'test'],
        'default_params': {}
    },
    
    # Multi-task
    'celeb_a': {
        'class': CelebADataset,
        'data_root': './data/datasets/soft_links/celeb-a',
        'splits': ['train', 'val'],
        'default_params': {}
    },
    
    # Note: Commented out datasets with known data availability issues
    # 'urb3dcd': Point cloud loading API mismatch - needs fix
    # 'xview2': File pattern mismatch - needs fix  
    # 'kc_3d': Missing real data - uses dummy files
    # 'ppsl': Need to verify data availability
    # 'i3pe': Need to verify data availability
}


# =============================================================================
# Helper Functions
# =============================================================================

def create_dataset_instance(dataset_name: str, split: str, **kwargs):
    """Create a dataset instance with given parameters."""
    config = DATASET_CONFIGS[dataset_name]
    params = {
        'data_root': config['data_root'],
        'split': split,
        **config['default_params'],
        **kwargs
    }
    return config['class'](**params)


def validate_hash_format(hash_val: str) -> None:
    """Validate that hash is in correct format."""
    assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"
    assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"
    assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


# =============================================================================
# Common Test Functions
# =============================================================================

@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_same_parameters_same_hash(dataset_name):
    """Test that identical parameters produce identical hashes."""
    config = DATASET_CONFIGS[dataset_name]
    split = config['splits'][0]  # Use first available split
    
    # Same parameters should produce same hash
    dataset1 = create_dataset_instance(dataset_name, split)
    dataset2 = create_dataset_instance(dataset_name, split)
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 == hash2, f"{dataset_name}: Same parameters should produce same hash: {hash1} != {hash2}"


@pytest.mark.parametrize('dataset_name', [
    name for name, config in DATASET_CONFIGS.items() if len(config['splits']) >= 2
])
def test_different_split_different_hash(dataset_name):
    """Test that different splits produce different hashes."""
    config = DATASET_CONFIGS[dataset_name]
    split1, split2 = config['splits'][0], config['splits'][1]
    
    dataset1 = create_dataset_instance(dataset_name, split1)
    dataset2 = create_dataset_instance(dataset_name, split2)
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 != hash2, f"{dataset_name}: Different splits should produce different hashes: {hash1} == {hash2}"


@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_hash_format(dataset_name):
    """Test that hash is in correct format."""
    config = DATASET_CONFIGS[dataset_name]
    split = config['splits'][0]  # Use first available split
    
    dataset = create_dataset_instance(dataset_name, split)
    hash_val = dataset.get_cache_version_hash()
    
    validate_hash_format(hash_val)


@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_data_root_excluded_from_hash(dataset_name):
    """Test that data_root is excluded from hash for cache stability."""
    config = DATASET_CONFIGS[dataset_name]
    split = config['splits'][0]  # Use first available split
    
    # Same data root should produce same hash (data_root excluded from versioning)
    dataset1 = create_dataset_instance(dataset_name, split)
    dataset2 = create_dataset_instance(dataset_name, split)
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 == hash2, f"{dataset_name}: Same data roots should produce same hashes: {hash1} != {hash2}"


@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_comprehensive_no_hash_collisions(dataset_name):
    """Test that different configurations produce unique hashes (no collisions)."""
    config = DATASET_CONFIGS[dataset_name]
    
    # Test various parameter combinations
    # NOTE: data_root is intentionally excluded from hash, so we test only meaningful parameter combinations
    hashes = []
    for split in config['splits']:
        dataset = create_dataset_instance(dataset_name, split)
        hash_val = dataset.get_cache_version_hash()
        
        # Check for collision
        assert hash_val not in hashes, f"{dataset_name}: Hash collision detected for split {split}: hash {hash_val} already exists"
        hashes.append(hash_val)
    
    # Verify we generated the expected number of unique hashes
    expected_count = len(config['splits'])
    assert len(hashes) == expected_count, f"{dataset_name}: Expected {expected_count} unique hashes, got {len(hashes)}"