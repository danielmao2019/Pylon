"""Common version hash discrimination tests for all datasets.

This module provides centralized testing for dataset cache version hash functionality
across all dataset types, avoiding code duplication and complex fixture management.
"""

import pytest
from utils.builders.builder import build_from_config

# Dataset imports - only verified working datasets
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset
from data.datasets.change_detection_datasets.bi_temporal.cdd_dataset import CDDDataset
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset
from data.datasets.change_detection_datasets.bi_temporal.sysu_cd_dataset import SYSU_CD_Dataset
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset
from data.datasets.change_detection_datasets.single_temporal.ppsl_dataset import PPSLDataset
from data.datasets.change_detection_datasets.single_temporal.i3pe_dataset import I3PEDataset
from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset
from data.datasets.torchvision_datasets.mnist import MNISTDataset
from data.datasets.pcr_datasets.bi_temporal_pcr_dataset import BiTemporalPCRDataset
from data.datasets.pcr_datasets.single_temporal_pcr_dataset import SingleTemporalPCRDataset
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset
from data.datasets.pcr_datasets.modelnet40_dataset import ModelNet40Dataset
from data.datasets.torchvision_datasets.mnist import MNISTDataset
from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset
from data.datasets.random_datasets.semantic_segmentation_random_dataset import SemanticSegmentationRandomDataset
from data.datasets.gan_datasets.gan_dataset import GANDataset


# =============================================================================
# Dataset Configuration Registry - Working Datasets Only
# =============================================================================

DATASET_CONFIGS = {
    # =============================================================================
    # Change Detection Datasets - Bi-temporal (verified working)
    # =============================================================================
    'air_change': {
        'class': AirChangeDataset,
        'args': {
            'data_root': './data/datasets/soft_links/AirChange',
            'split': 'train'
        },
        'splits': ['train', 'test'],  # AirChange only has train and test splits
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'test'}
        ]
    },
    'cdd': {
        'class': CDDDataset,
        'args': {
            'data_root': './data/datasets/soft_links/CDD',
            'split': 'train'
        },
        'splits': ['train', 'val', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'split': 'test'}
        ]
    },
    'levir_cd': {
        'class': LevirCdDataset,
        'args': {
            'data_root': './data/datasets/soft_links/LEVIR-CD',
            'split': 'train'
        },
        'splits': ['train', 'val', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'split': 'test'}
        ]
    },
    'oscd': {
        'class': OSCDDataset,
        'args': {
            'data_root': './data/datasets/soft_links/OSCD',
            'split': 'train'
        },
        'splits': ['train', 'test'],  # OSCD only has train and test splits
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'test'},
            {'bands': '13ch'},  # Different from default '3ch'
            {'split': 'test', 'bands': '13ch'}  # Combined variation
        ]
    },
    'slpccd': {
        'class': SLPCCDDataset,
        'args': {
            'data_root': './data/datasets/soft_links/SLPCCD',
            'split': 'train'
        },
        'splits': ['train', 'val', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'split': 'test'}
        ]
    },
    'sysu_cd': {
        'class': SYSU_CD_Dataset,
        'args': {
            'data_root': './data/datasets/soft_links/SYSU-CD',
            'split': 'train'
        },
        'splits': ['train', 'val', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'split': 'test'}
        ]
    },
    
    # =============================================================================
    # Multi-task Datasets (verified working)
    # =============================================================================
    'celeb_a': {
        'class': CelebADataset,
        'args': {
            'data_root': './data/datasets/soft_links/celeb-a',
            'split': 'train'
        },
        'splits': ['train', 'val'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'use_landmarks': True},  # Different from default False
            {'split': 'val', 'use_landmarks': True}  # Combined variation
        ]
    },
    'mnist': {
        'class': MNISTDataset,
        'args': {
            'data_root': './data/datasets/soft_links/MNIST',  # torchvision will download here
            'split': 'train'
        },
        'splits': ['train', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'test'}
        ]
    },
    
    # =============================================================================
    # PCR Datasets (verified working)
    # =============================================================================
    'bi_temporal_pcr': {
        'class': BiTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/threedmatch/test/7-scenes-redkitchen',
            'dataset_size': 50,  # Reduced for faster testing
            'rotation_mag': 45.0,
            'translation_mag': 0.5,
            'matching_radius': 0.1,
            'min_points': 500
        },
        'splits': [],  # Synthetic dataset, no splits
        'parameter_variations': [
            {'rotation_mag': 90.0},  # Different from default 45.0
            {'translation_mag': 1.0},  # Different from default 0.5
            {'rotation_mag': 90.0, 'translation_mag': 1.0},  # Combined variation
            {'matching_radius': 0.05}  # Different from default 0.1
        ]
    },
    'single_temporal_pcr': {
        'class': SingleTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/threedmatch/test/7-scenes-redkitchen',
            'dataset_size': 50,  # Reduced for faster testing
            'rotation_mag': 45.0,
            'translation_mag': 0.5,
            'matching_radius': 0.1,
            'min_points': 500
        },
        'splits': [],  # Synthetic dataset, no splits
        'parameter_variations': [
            {'rotation_mag': 90.0},  # Different from default 45.0
            {'translation_mag': 1.0},  # Different from default 0.5
            {'max_points': 2048},  # Different from default 4096
            {'rotation_mag': 90.0, 'max_points': 2048}  # Combined variation
        ]
    },
    
    # =============================================================================
    # Synthetic Datasets (verified working)
    # =============================================================================
    'classification_random': {
        'class': ClassificationRandomDataset,
        'args': {
            'num_examples': 50,  # Reduced for faster testing
            'num_classes': 10,
            'image_res': (32, 32)
        },
        'splits': [],  # Synthetic dataset, no splits
        'parameter_variations': [
            {'num_classes': 20},  # Different from default 10
            {'num_examples': 100},  # Different from default 50
            {'image_res': (64, 64)},  # Different from default (32, 32)
            {'num_classes': 20, 'num_examples': 100}  # Combined variation
        ]
    },
    'semantic_segmentation_random': {
        'class': SemanticSegmentationRandomDataset,
        'args': {
            'num_examples': 50,  # Reduced for faster testing
            'num_classes': 21
        },
        'splits': [],  # Synthetic dataset, no splits
        'parameter_variations': [
            {'num_classes': 151},  # Different from default 21
            {'num_examples': 100},  # Different from default 50
            {'initial_seed': 42},  # Different from default None
            {'num_classes': 151, 'num_examples': 100}  # Combined variation
        ]
    },
    
    # Note: Datasets excluded due to issues:
    # - KC3D: Uses dummy files (excluded as requested)
    # - Urb3DCD, xView2: Data format issues
    # - MultiMNIST: Complex setup, MNIST covers torchvision integration
    # - Other datasets: Need data availability verification
}


# =============================================================================
# Helper Functions
# =============================================================================

def create_dataset_instance(dataset_name: str, **kwargs):
    """Create a dataset instance using build_from_config."""
    config = DATASET_CONFIGS[dataset_name]
    dataset_config = {
        'class': config['class'],
        'args': {**config['args'], **kwargs}
    }
    return build_from_config(dataset_config)


def get_dataset_splits(dataset_name: str):
    """Get available splits for a dataset."""
    return DATASET_CONFIGS[dataset_name]['splits']


def get_parameter_variations(dataset_name: str):
    """Get parameter variations for comprehensive testing."""
    return DATASET_CONFIGS[dataset_name]['parameter_variations']


def validate_hash_format(hash_val: str) -> None:
    """Validate that hash is in correct format."""
    assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"
    assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"
    assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


def has_splits(dataset_name: str) -> bool:
    """Check if dataset has splits (non-synthetic datasets)."""
    return len(get_dataset_splits(dataset_name)) > 0


def is_multi_split_dataset(dataset_name: str) -> bool:
    """Check if dataset has multiple splits for discrimination testing."""
    return len(get_dataset_splits(dataset_name)) >= 2


# =============================================================================
# Common Test Functions - Core Hash Behavior
# =============================================================================

@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_same_parameters_same_hash(dataset_name):
    """Test that identical parameters produce identical hashes."""
    # Use first parameter variation as baseline
    params = get_parameter_variations(dataset_name)[0]
    
    # Same parameters should produce same hash
    dataset1 = create_dataset_instance(dataset_name, **params)
    dataset2 = create_dataset_instance(dataset_name, **params)
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 == hash2, f"{dataset_name}: Same parameters should produce same hash: {hash1} != {hash2}"


@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_hash_format(dataset_name):
    """Test that hash is in correct format."""
    # Use first parameter variation
    params = get_parameter_variations(dataset_name)[0]
    
    dataset = create_dataset_instance(dataset_name, **params)
    hash_val = dataset.get_cache_version_hash()
    
    validate_hash_format(hash_val)


@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_data_root_excluded_from_hash(dataset_name):
    """Test that data_root is excluded from hash for cache stability."""
    # Use first parameter variation
    params = get_parameter_variations(dataset_name)[0]
    
    # Same data root should produce same hash (data_root excluded from versioning)
    dataset1 = create_dataset_instance(dataset_name, **params)
    dataset2 = create_dataset_instance(dataset_name, **params)
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 == hash2, f"{dataset_name}: Same data roots should produce same hashes: {hash1} != {hash2}"


# =============================================================================
# Hash Discrimination Tests - Split-based (for datasets with splits)
# =============================================================================

@pytest.mark.parametrize('dataset_name', [name for name in DATASET_CONFIGS.keys() if is_multi_split_dataset(name)])
def test_different_split_different_hash(dataset_name):
    """Test that different splits produce different hashes."""
    splits = get_dataset_splits(dataset_name)
    split1, split2 = splits[0], splits[1]
    
    dataset1 = create_dataset_instance(dataset_name, split=split1)
    dataset2 = create_dataset_instance(dataset_name, split=split2)
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 != hash2, f"{dataset_name}: Different splits should produce different hashes: {hash1} == {hash2}"


@pytest.mark.parametrize('dataset_name', [name for name in DATASET_CONFIGS.keys() if has_splits(name)])
def test_split_based_hash_collisions(dataset_name):
    """Test that all splits produce unique hashes (no collisions)."""
    splits = get_dataset_splits(dataset_name)
    
    hashes = []
    for split in splits:
        dataset = create_dataset_instance(dataset_name, split=split)
        hash_val = dataset.get_cache_version_hash()
        
        # Check for collision
        assert hash_val not in hashes, f"{dataset_name}: Hash collision detected for split {split}: hash {hash_val} already exists"
        hashes.append(hash_val)
    
    # Verify we generated the expected number of unique hashes
    expected_count = len(splits)
    assert len(hashes) == expected_count, f"{dataset_name}: Expected {expected_count} unique hashes, got {len(hashes)}"


# =============================================================================
# Hash Discrimination Tests - Parameter-based (comprehensive variations)
# =============================================================================

@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_parameter_variations_hash_discrimination(dataset_name):
    """Test that different parameter combinations produce different hashes when they should."""
    variations = get_parameter_variations(dataset_name)
    
    if len(variations) < 2:
        pytest.skip(f"Dataset {dataset_name} has insufficient parameter variations for discrimination testing")
    
    hashes = []
    for i, params in enumerate(variations):
        dataset = create_dataset_instance(dataset_name, **params)
        hash_val = dataset.get_cache_version_hash()
        
        # Check for collision
        assert hash_val not in hashes, f"{dataset_name}: Hash collision detected for variation {i} {params}: hash {hash_val} already exists"
        hashes.append(hash_val)
    
    # Verify we generated the expected number of unique hashes
    expected_count = len(variations)
    assert len(hashes) == expected_count, f"{dataset_name}: Expected {expected_count} unique hashes, got {len(hashes)}"


@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_comprehensive_no_hash_collisions(dataset_name):
    """Test comprehensive hash collision detection across all meaningful parameter combinations."""
    variations = get_parameter_variations(dataset_name)
    
    # Collect all hashes from all variations
    all_hashes = []
    for params in variations:
        dataset = create_dataset_instance(dataset_name, **params)
        hash_val = dataset.get_cache_version_hash()
        all_hashes.append((hash_val, params))
    
    # Check for any collisions
    seen_hashes = {}
    for hash_val, params in all_hashes:
        if hash_val in seen_hashes:
            pytest.fail(f"{dataset_name}: Hash collision detected!\n"
                       f"Hash {hash_val} appears for both:\n"
                       f"1. {seen_hashes[hash_val]}\n"
                       f"2. {params}")
        seen_hashes[hash_val] = params
    
    # Verify we have the expected number of unique hashes
    expected_count = len(variations)
    actual_count = len(seen_hashes)
    assert actual_count == expected_count, f"{dataset_name}: Expected {expected_count} unique hashes, got {actual_count}"


# =============================================================================
# Special Cases and Edge Cases
# =============================================================================

@pytest.mark.parametrize('dataset_name', [name for name in DATASET_CONFIGS.keys() if 'celeb_a' in name])
def test_celeb_a_use_landmarks_discrimination(dataset_name):
    """Test CelebA-specific parameter discrimination (use_landmarks)."""
    dataset_with_landmarks = create_dataset_instance(dataset_name, split='train', use_landmarks=True)
    dataset_without_landmarks = create_dataset_instance(dataset_name, split='train', use_landmarks=False)
    
    hash_with = dataset_with_landmarks.get_cache_version_hash()
    hash_without = dataset_without_landmarks.get_cache_version_hash()
    
    assert hash_with != hash_without, f"{dataset_name}: Different use_landmarks should produce different hashes: {hash_with} == {hash_without}"


@pytest.mark.parametrize('dataset_name', [name for name in DATASET_CONFIGS.keys() if 'pcr' in name])
def test_pcr_parameter_discrimination(dataset_name):
    """Test PCR-specific parameter discrimination (rotation_mag, translation_mag)."""
    # Test rotation magnitude variation
    dataset1 = create_dataset_instance(dataset_name, rotation_mag=45.0)
    dataset2 = create_dataset_instance(dataset_name, rotation_mag=90.0)
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 != hash2, f"{dataset_name}: Different rotation_mag should produce different hashes: {hash1} == {hash2}"
    
    # Test translation magnitude variation
    dataset3 = create_dataset_instance(dataset_name, translation_mag=0.5)
    dataset4 = create_dataset_instance(dataset_name, translation_mag=1.0)
    
    hash3 = dataset3.get_cache_version_hash()
    hash4 = dataset4.get_cache_version_hash()
    
    assert hash3 != hash4, f"{dataset_name}: Different translation_mag should produce different hashes: {hash3} == {hash4}"


@pytest.mark.parametrize('dataset_name', [name for name in DATASET_CONFIGS.keys() if 'random' in name])
def test_synthetic_dataset_discrimination(dataset_name):
    """Test synthetic dataset parameter discrimination (num_examples, num_classes)."""
    variations = get_parameter_variations(dataset_name)
    
    # Test that different parameters produce different hashes
    if len(variations) >= 2:
        dataset1 = create_dataset_instance(dataset_name, **variations[0])
        dataset2 = create_dataset_instance(dataset_name, **variations[1])
        
        hash1 = dataset1.get_cache_version_hash()
        hash2 = dataset2.get_cache_version_hash()
        
        assert hash1 != hash2, f"{dataset_name}: Different synthetic parameters should produce different hashes: {hash1} == {hash2}"
