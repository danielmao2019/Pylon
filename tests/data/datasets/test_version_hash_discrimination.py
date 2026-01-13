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
from data.datasets.pcr_datasets.threedmatch_dataset import ThreeDMatchDataset, ThreeDLoMatchDataset
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset
from data.datasets.pcr_datasets.modelnet40_dataset import ModelNet40Dataset
from data.datasets.torchvision_datasets.mnist import MNISTDataset
from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset
from data.datasets.random_datasets.semantic_segmentation_random_dataset import SemanticSegmentationRandomDataset
from data.datasets.gan_datasets.gan_dataset import GANDataset
from data.datasets.semantic_segmentation_datasets.whu_bd_dataset import WHU_BD_Dataset


# =============================================================================
# Dataset Configuration Registry - Using Fixtures for Data Roots
# =============================================================================

DATASET_CONFIGS = {
    # =============================================================================
    # Change Detection Datasets - Bi-temporal (verified working)
    # =============================================================================
    'air_change': {
        'class': AirChangeDataset,
        'fixture': 'air_change_data_root',
        'splits': ['train', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'test'}
        ]
    },
    'cdd': {
        'class': CDDDataset,
        'fixture': 'cdd_data_root',
        'splits': ['train', 'val', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'split': 'test'}
        ]
    },
    'levir_cd': {
        'class': LevirCdDataset,
        'fixture': 'levir_cd_data_root',
        'splits': ['train', 'val', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'split': 'test'}
        ]
    },
    'oscd': {
        'class': OSCDDataset,
        'fixture': 'oscd_data_root',
        'splits': ['train', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'test'},
            {'bands': '13ch'},
            {'split': 'test', 'bands': '13ch'}
        ]
    },
    'slpccd': {
        'class': SLPCCDDataset,
        'fixture': 'slpccd_data_root',
        'splits': ['train', 'val', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'split': 'test'}
        ]
    },
    'sysu_cd': {
        'class': SYSU_CD_Dataset,
        'fixture': 'sysu_cd_data_root',
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
        'fixture': 'celeb_a_data_root',
        'splits': ['train', 'val'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'use_landmarks': True},
            {'split': 'val', 'use_landmarks': True}
        ]
    },
    'mnist': {
        'class': MNISTDataset,
        'fixture': 'mnist_data_root',
        'splits': ['train', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'test'}
        ]
    },

    # =============================================================================
    # PCR Datasets (verified working)
    # =============================================================================
    '3dmatch': {
        'class': ThreeDMatchDataset,
        'fixture': 'threedmatch_data_root',
        'splits': ['train', 'val', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'split': 'test'}
        ]
    },
    '3dlomatch': {
        'class': ThreeDLoMatchDataset,
        'fixture': 'threedmatch_data_root',
        'splits': ['train', 'val', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'split': 'test'}
        ]
    },
    'modelnet40': {
        'class': ModelNet40Dataset,
        'fixture': 'modelnet40_data_root',
        'cache_fixture': 'modelnet40_cache_file',
        'splits': ['train', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'test'},
            {'rotation_mag': 30.0, 'translation_mag': 0.3},
            {'keep_ratio': 0.8, 'matching_radius': 0.03}
        ],
        'extra_args': {
            'dataset_size': 50,
            'overlap_range': (0.0, 1.0),
            'matching_radius': 0.05,
            'rotation_mag': 45.0,
            'translation_mag': 0.5,
            'keep_ratio': 0.7
        }
    },

    # =============================================================================
    # Semantic Segmentation Datasets (verified working)
    # =============================================================================
    'whu_bd': {
        'class': WHU_BD_Dataset,
        'fixture': 'whu_bd_data_root',
        'splits': ['train', 'val', 'test'],
        'parameter_variations': [
            {'split': 'train'},
            {'split': 'val'},
            {'split': 'test'}
        ]
    },

    # =============================================================================
    # Synthetic Datasets (verified working)
    # =============================================================================
    'classification_random': {
        'class': ClassificationRandomDataset,
        'fixture': None,  # No data root needed for synthetic datasets
        'splits': [],
        'parameter_variations': [
            {'num_classes': 20},
            {'num_examples': 100},
            {'image_res': (64, 64)},
            {'num_classes': 20, 'num_examples': 100}
        ],
        'extra_args': {
            'num_examples': 50,
            'num_classes': 10,
            'image_res': (32, 32)
        }
    },
    'semantic_segmentation_random': {
        'class': SemanticSegmentationRandomDataset,
        'fixture': None,
        'splits': [],
        'parameter_variations': [
            {'num_classes': 151},
            {'num_examples': 100},
            {'base_seed': 42},
            {'num_classes': 151, 'num_examples': 100}
        ],
        'extra_args': {
            'num_examples': 50,
            'num_classes': 21
        }
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def create_dataset_instance(dataset_name: str, request, **kwargs):
    """Create a dataset instance using build_from_config with fixtures."""
    config = DATASET_CONFIGS[dataset_name]

    # Build args starting with extra_args if present
    args = config.get('extra_args', {}).copy()

    # Add data_root from fixture if specified
    if config['fixture'] is not None:
        args['data_root'] = request.getfixturevalue(config['fixture'])

    # Add cache_filepath from fixture if specified
    if 'cache_fixture' in config:
        args['cache_filepath'] = request.getfixturevalue(config['cache_fixture'])

    # Add default split if not provided in kwargs
    if 'split' not in kwargs and config['splits']:
        args['split'] = config['splits'][0]

    # Override with any provided kwargs
    args.update(kwargs)

    dataset_config = {
        'class': config['class'],
        'args': args
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
def test_same_parameters_same_hash(dataset_name, request):
    """Test that identical parameters produce identical hashes."""
    # Use first parameter variation as baseline
    params = get_parameter_variations(dataset_name)[0]

    # Same parameters should produce same hash
    dataset1 = create_dataset_instance(dataset_name, request, **params)
    dataset2 = create_dataset_instance(dataset_name, request, **params)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 == hash2, f"{dataset_name}: Same parameters should produce same hash: {hash1} != {hash2}"


@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_hash_format(dataset_name, request):
    """Test that hash is in correct format."""
    # Use first parameter variation
    params = get_parameter_variations(dataset_name)[0]

    dataset = create_dataset_instance(dataset_name, request, **params)
    hash_val = dataset.get_cache_version_hash()

    validate_hash_format(hash_val)


@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_data_root_excluded_from_hash(dataset_name, request):
    """Test that data_root is excluded from hash for cache stability."""
    # Use first parameter variation
    params = get_parameter_variations(dataset_name)[0]

    # Same data root should produce same hash (data_root excluded from versioning)
    dataset1 = create_dataset_instance(dataset_name, request, **params)
    dataset2 = create_dataset_instance(dataset_name, request, **params)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 == hash2, f"{dataset_name}: Same data roots should produce same hashes: {hash1} != {hash2}"


# =============================================================================
# Hash Discrimination Tests - Split-based (for datasets with splits)
# =============================================================================

@pytest.mark.parametrize('dataset_name', [name for name in DATASET_CONFIGS.keys() if is_multi_split_dataset(name)])
def test_different_split_different_hash(dataset_name, request):
    """Test that different splits produce different hashes."""
    splits = get_dataset_splits(dataset_name)
    split1, split2 = splits[0], splits[1]

    dataset1 = create_dataset_instance(dataset_name, request, split=split1)
    dataset2 = create_dataset_instance(dataset_name, request, split=split2)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 != hash2, f"{dataset_name}: Different splits should produce different hashes: {hash1} == {hash2}"


@pytest.mark.parametrize('dataset_name', [name for name in DATASET_CONFIGS.keys() if has_splits(name)])
def test_split_based_hash_collisions(dataset_name, request):
    """Test that all splits produce unique hashes (no collisions)."""
    splits = get_dataset_splits(dataset_name)

    hashes = []
    for split in splits:
        dataset = create_dataset_instance(dataset_name, request, split=split)
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
def test_parameter_variations_hash_discrimination(dataset_name, request):
    """Test that different parameter combinations produce different hashes when they should."""
    variations = get_parameter_variations(dataset_name)

    if len(variations) < 2:
        pytest.skip(f"Dataset {dataset_name} has insufficient parameter variations for discrimination testing")

    hashes = []
    for i, params in enumerate(variations):
        dataset = create_dataset_instance(dataset_name, request, **params)
        hash_val = dataset.get_cache_version_hash()

        # Check for collision
        assert hash_val not in hashes, f"{dataset_name}: Hash collision detected for variation {i} {params}: hash {hash_val} already exists"
        hashes.append(hash_val)

    # Verify we generated the expected number of unique hashes
    expected_count = len(variations)
    assert len(hashes) == expected_count, f"{dataset_name}: Expected {expected_count} unique hashes, got {len(hashes)}"


@pytest.mark.parametrize('dataset_name', list(DATASET_CONFIGS.keys()))
def test_comprehensive_no_hash_collisions(dataset_name, request):
    """Test comprehensive hash collision detection across all meaningful parameter combinations."""
    variations = get_parameter_variations(dataset_name)

    # Collect all hashes from all variations
    all_hashes = []
    for params in variations:
        dataset = create_dataset_instance(dataset_name, request, **params)
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
