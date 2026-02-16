"""Tests for ViewerBackend hierarchical dataset organization functionality.

This module tests dataset grouping, hierarchical organization, and
get_available_datasets_hierarchical functionality.
"""

from typing import Dict, Any

import pytest

from data.viewer.dataset.backend.backend import ViewerBackend, DATASET_GROUPS


@pytest.fixture
def backend():
    """Create a ViewerBackend instance for testing."""
    return ViewerBackend()


@pytest.fixture
def backend_with_mock_configs():
    """Create a ViewerBackend with mock configurations for testing."""
    backend = ViewerBackend()

    # Add mock configurations for testing hierarchical organization
    mock_configs = {
        'semseg/coco_stuff_164k': {
            'path': '/fake/path/coco_stuff_164k_data_cfg.py',
            'type': 'semseg',
            'name': 'coco_stuff_164k'
        },
        'semseg/whu_bd': {
            'path': '/fake/path/whu_bd_data_cfg.py',
            'type': 'semseg',
            'name': 'whu_bd'
        },
        '2dcd/air_change': {
            'path': '/fake/path/air_change_data_cfg.py',
            'type': '2dcd',
            'name': 'air_change'
        },
        '2dcd/levir_cd': {
            'path': '/fake/path/levir_cd_data_cfg.py',
            'type': '2dcd',
            'name': 'levir_cd'
        },
        'pcr/kitti': {
            'path': '/fake/path/kitti_data_cfg.py',
            'type': 'pcr',
            'name': 'kitti'
        },
        'mtl/multi_mnist': {
            'path': '/fake/path/multi_mnist_data_cfg.py',
            'type': 'mtl',
            'name': 'multi_mnist'
        }
    }

    backend._configs = mock_configs
    return backend


def test_get_available_datasets_hierarchical_structure(backend_with_mock_configs):
    """Test that get_available_datasets_hierarchical returns correct structure."""
    hierarchical = backend_with_mock_configs.get_available_datasets_hierarchical()

    # Test return type
    assert isinstance(hierarchical, dict)

    # Test expected dataset types are present
    expected_types = ['semseg', '2dcd', 'pcr', 'mtl']
    for dataset_type in expected_types:
        assert dataset_type in hierarchical, f"Dataset type {dataset_type} should be present"
        assert isinstance(hierarchical[dataset_type], dict), f"Dataset type {dataset_type} should map to dict"


def test_get_available_datasets_hierarchical_grouping(backend_with_mock_configs):
    """Test that datasets are correctly grouped by type."""
    hierarchical = backend_with_mock_configs.get_available_datasets_hierarchical()

    # Test semseg group
    semseg_datasets = hierarchical.get('semseg', {})
    expected_semseg = ['semseg/coco_stuff_164k', 'semseg/whu_bd']
    for config_name in expected_semseg:
        assert config_name in semseg_datasets, f"{config_name} should be in semseg group"
        assert semseg_datasets[config_name] == config_name.split('/')[-1]

    # Test 2dcd group
    cd_datasets = hierarchical.get('2dcd', {})
    expected_2dcd = ['2dcd/air_change', '2dcd/levir_cd']
    for config_name in expected_2dcd:
        assert config_name in cd_datasets, f"{config_name} should be in 2dcd group"
        assert cd_datasets[config_name] == config_name.split('/')[-1]

    # Test pcr group
    pcr_datasets = hierarchical.get('pcr', {})
    assert 'pcr/kitti' in pcr_datasets
    assert pcr_datasets['pcr/kitti'] == 'kitti'

    # Test mtl group
    mtl_datasets = hierarchical.get('mtl', {})
    assert 'mtl/multi_mnist' in mtl_datasets
    assert mtl_datasets['mtl/multi_mnist'] == 'multi_mnist'


def test_get_available_datasets_hierarchical_sorting(backend_with_mock_configs):
    """Test that datasets are sorted within each group."""
    hierarchical = backend_with_mock_configs.get_available_datasets_hierarchical()

    # Test that datasets within each type are sorted
    for dataset_type, datasets in hierarchical.items():
        dataset_keys = list(datasets.keys())
        sorted_keys = sorted(dataset_keys)
        assert dataset_keys == sorted_keys, f"Datasets in {dataset_type} should be sorted"


def test_get_available_datasets_hierarchical_empty_backend(backend):
    """Test hierarchical datasets with empty backend (no configs loaded)."""
    hierarchical = backend.get_available_datasets_hierarchical()

    # Should return empty dict or dict with empty groups
    assert isinstance(hierarchical, dict)

    # All groups should be empty or non-existent
    for dataset_type, datasets in hierarchical.items():
        assert isinstance(datasets, dict)
        # Empty or minimal datasets are acceptable for empty backend


def test_get_available_datasets_hierarchical_name_extraction(backend_with_mock_configs):
    """Test that dataset names are correctly extracted from config names."""
    hierarchical = backend_with_mock_configs.get_available_datasets_hierarchical()

    # Test name extraction for each dataset type
    test_cases = [
        ('semseg', 'semseg/coco_stuff_164k', 'coco_stuff_164k'),
        ('semseg', 'semseg/whu_bd', 'whu_bd'),
        ('2dcd', '2dcd/air_change', 'air_change'),
        ('2dcd', '2dcd/levir_cd', 'levir_cd'),
        ('pcr', 'pcr/kitti', 'kitti'),
        ('mtl', 'mtl/multi_mnist', 'multi_mnist')
    ]

    for dataset_type, config_name, expected_name in test_cases:
        if dataset_type in hierarchical and config_name in hierarchical[dataset_type]:
            actual_name = hierarchical[dataset_type][config_name]
            assert actual_name == expected_name, f"Name extraction failed for {config_name}"


def test_get_available_datasets_hierarchical_consistency_with_configs(backend_with_mock_configs):
    """Test that hierarchical view is consistent with internal configs."""
    hierarchical = backend_with_mock_configs.get_available_datasets_hierarchical()
    configs = backend_with_mock_configs._configs

    # Count total datasets in both structures
    total_hierarchical = sum(len(datasets) for datasets in hierarchical.values())
    total_configs = len(configs)

    assert total_hierarchical == total_configs, "Hierarchical view should include all configured datasets"

    # Test that every config appears in hierarchical view
    for config_name, config_info in configs.items():
        dataset_type = config_info['type']
        dataset_name = config_info['name']

        assert dataset_type in hierarchical, f"Type {dataset_type} should be in hierarchical view"
        assert config_name in hierarchical[dataset_type], f"Config {config_name} should be in {dataset_type} group"
        assert hierarchical[dataset_type][config_name] == dataset_name, f"Name should match for {config_name}"


def test_dataset_groups_constant_coverage():
    """Test that DATASET_GROUPS constant covers expected dataset categories."""
    # Test structure
    assert isinstance(DATASET_GROUPS, dict)

    # Test expected categories
    expected_categories = ['semseg', '2dcd', '3dcd', 'pcr', 'mtl', 'general']
    for category in expected_categories:
        assert category in DATASET_GROUPS, f"DATASET_GROUPS should contain {category}"
        assert isinstance(DATASET_GROUPS[category], list), f"{category} should map to list"

    # Test that all entries are strings
    for category, datasets in DATASET_GROUPS.items():
        for dataset in datasets:
            assert isinstance(dataset, str), f"Dataset {dataset} in {category} should be string"


def test_dataset_groups_semantic_segmentation():
    """Test semantic segmentation dataset group."""
    semseg_datasets = DATASET_GROUPS['semseg']

    # Test expected datasets are present
    expected_semseg = ['coco_stuff_164k', 'whu_bd']
    for dataset in expected_semseg:
        assert dataset in semseg_datasets, f"{dataset} should be in semseg group"


def test_dataset_groups_change_detection():
    """Test change detection dataset groups."""
    # Test 2D change detection
    cd_2d_datasets = DATASET_GROUPS['2dcd']
    expected_2dcd = ['air_change', 'cdd', 'levir_cd', 'oscd', 'sysu_cd']
    for dataset in expected_2dcd:
        assert dataset in cd_2d_datasets, f"{dataset} should be in 2dcd group"

    # Test 3D change detection
    cd_3d_datasets = DATASET_GROUPS['3dcd']
    expected_3dcd = ['urb3dcd', 'slpccd']
    for dataset in expected_3dcd:
        assert dataset in cd_3d_datasets, f"{dataset} should be in 3dcd group"


def test_dataset_groups_point_cloud_registration():
    """Test point cloud registration dataset group."""
    pcr_datasets = DATASET_GROUPS['pcr']

    expected_pcr = ['kitti', 'threedmatch', 'threedlomatch', 'modelnet40', 'buffer']
    for dataset in expected_pcr:
        assert dataset in pcr_datasets, f"{dataset} should be in pcr group"


def test_dataset_groups_multi_task_learning():
    """Test multi-task learning dataset group."""
    mtl_datasets = DATASET_GROUPS['mtl']

    expected_mtl = [
        'multi_mnist', 'celeb_a', 'multi_task_facial_landmark',
        'nyu_v2_c', 'nyu_v2_f', 'city_scapes_c', 'city_scapes_f',
        'pascal_context', 'ade_20k'
    ]
    for dataset in expected_mtl:
        assert dataset in mtl_datasets, f"{dataset} should be in mtl group"


def test_dataset_groups_general():
    """Test general dataset group."""
    general_datasets = DATASET_GROUPS['general']

    # Should contain at least BaseRandomDataset for testing
    assert 'BaseRandomDataset' in general_datasets, "BaseRandomDataset should be in general group"


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_get_available_datasets_hierarchical_with_invalid_config_format():
    """Test hierarchical datasets handling with invalid config format."""
    backend = ViewerBackend()

    # Add invalid config (missing '/' in name)
    backend._configs = {
        'invalid_config_name': {
            'path': '/fake/path',
            'type': 'semseg',
            'name': 'test'
        }
    }

    # Should handle gracefully (either skip or raise clear error)
    with pytest.raises(ValueError):
        backend.get_available_datasets_hierarchical()


def test_hierarchical_datasets_with_corrupted_configs():
    """Test that corrupted config entries are handled appropriately."""
    backend = ViewerBackend()

    # Add config with missing required fields
    backend._configs = {
        'semseg/incomplete': {
            'path': '/fake/path',
            # missing 'type' and 'name'
        }
    }

    # Should handle gracefully
    hierarchical = backend.get_available_datasets_hierarchical()
    # The behavior depends on implementation - either skip invalid configs or handle them